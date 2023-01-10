import sys
_module = sys.modules[__name__]
del sys
download_models = _module
neural_dream = _module
CaffeLoader = _module
dream_auto = _module
dream_experimental = _module
dream_image = _module
dream_model = _module
dream_tile = _module
dream_utils = _module
helper_layers = _module
loss_layers = _module
models = _module
googlenet = _module
bvlc_googlenet = _module
googlenet_cars = _module
googlenet_layer_names = _module
googlenet_sos = _module
googlenetplaces = _module
inception = _module
inception5h = _module
inception_layer_names = _module
inceptionv3_keras = _module
resnet = _module
resnet_50_1by2_nsfw = _module
resnet_layer_names = _module

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


from collections import OrderedDict


from torch.utils.model_zoo import load_url


import copy


import random


import torch.nn as nn


import torch.optim as optim


import torchvision.transforms as transforms


import torchvision


from torchvision import models


import math


import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))


class VGG_SOD(nn.Module):

    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 100))


class VGG_FCN32S(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, (7, 7)), nn.ReLU(True), nn.Dropout(0.5), nn.Conv2d(4096, 4096, (1, 1)), nn.ReLU(True), nn.Dropout(0.5))


class VGG_PRUNED(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5))


class NIN(nn.Module):

    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        self.features = nn.Sequential(nn.Conv2d(3, 96, (11, 11), (4, 4)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Dropout(0.5), nn.Conv2d(384, 1024, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1000, (1, 1)), nn.ReLU(inplace=True), nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True), nn.Softmax())


class ModelParallel(nn.Module):

    def __init__(self, net, device_ids, device_splits):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(','))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(',')))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != 'c':
                device_list.append('cuda:' + str(device))
            else:
                device_list.append('cpu')
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != '':
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk
        return chunks

    def c(self, input, i):
        if input.type() == 'torch.FloatTensor' and 'cuda' in self.device_list[i]:
            input = input.type('torch.cuda.FloatTensor')
        elif input.type() == 'torch.cuda.FloatTensor' and 'cpu' in self.device_list[i]:
            input = input.type('torch.FloatTensor')
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1)
            else:
                input = chunk(input)
        return input


class Flatten(nn.Module):

    def forward(self, input):
        return torch.flatten(input, 1)


class ChannelMask(torch.nn.Module):

    def __init__(self, mode, channels=-1, rank_mode='norm', channel_percent=-1):
        super(ChannelMask, self).__init__()
        self.mode = mode
        self.channels = channels
        self.rank_mode = rank_mode
        self.channel_percent = channel_percent

    def list_channels(self, input):
        if input.is_cuda:
            channel_list = torch.zeros(input.size(1), device=input.get_device())
        else:
            channel_list = torch.zeros(input.size(1))
        for i in range(input.size(1)):
            y = input.clone().narrow(1, i, 1)
            if self.rank_mode == 'norm':
                y = torch.norm(y)
            elif self.rank_mode == 'sum':
                y = torch.sum(y)
            elif self.rank_mode == 'mean':
                y = torch.mean(y)
            elif self.rank_mode == 'norm-abs':
                y = torch.norm(torch.abs(y))
            elif self.rank_mode == 'sum-abs':
                y = torch.sum(torch.abs(y))
            elif self.rank_mode == 'mean-abs':
                y = torch.mean(torch.abs(y))
            channel_list[i] = y
        return channel_list

    def channel_strengths(self, input, num_channels):
        channel_list = self.list_channels(input)
        channels, idx = torch.sort(channel_list, 0, True)
        selected_channels = []
        for i in range(num_channels):
            if i < input.size(1):
                selected_channels.append(idx[i])
        return selected_channels

    def mask_threshold(self, input, channels, ft=1, fm=0.2, fw=5):
        t = torch.ones_like(input.squeeze(0)) * ft
        m = torch.ones_like(input.squeeze(0)) * fm
        for c in channels:
            m[c] = fw
        return (t * m).unsqueeze(0)

    def average_mask(self, input, channels):
        mask = torch.ones_like(input.squeeze(0))
        avg = torch.sum(channels) / input.size(1)
        for i in range(channels.size(1)):
            w = avg / channels[i]
            mask[i] = w
        return mask.unsqueeze(0)

    def average_tensor(self, input):
        channel_list = self.list_channels(input)
        mask = self.average_mask(input, channel_list)
        self.mask = mask

    def weak_channels(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=2, fw=0.2)
        self.mask = mask

    def strong_channels(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=0.2, fw=5)
        self.mask = mask

    def zero_weak(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=0, fw=1)
        self.mask = mask

    def mask_input(self, input):
        if self.channel_percent > 0:
            channels = int(float(self.channel_percent) / 100 * float(input.size(1)))
            if channels < input.size(1) and channels > 0:
                self.channels = channels
            else:
                self.channels = input.size(1)
        if self.mode == 'weak':
            input = self.weak_channels(input)
        elif self.mode == 'strong':
            input = self.strong_channels(input)
        elif self.mode == 'average':
            input = self.average_tensor(input)
        elif self.mode == 'zero_weak':
            input = self.zero_weak(input)

    def capture(self, input):
        self.mask_input(input)

    def forward(self, input):
        return self.mask * input


def percentile_zero(input, p=99.98, mode='norm'):
    if 'norm' in mode:
        px = input.norm(1)
    elif 'sum' in mode:
        px = input.sum(1)
    elif 'mean' in mode:
        px = input.sum(1)
    if 'abs' in mode:
        px, tp = dream_utils.tensor_percentile(abs(px), p), abs(px)
    else:
        tp = dream_utils.tensor_percentile(px, p)
    th = 0.01 * tp
    if 'abs' in mode:
        input[abs(input) < abs(th)] = 0
    else:
        input[input < th] = 0
    return input


class ChannelMod(torch.nn.Module):

    def __init__(self, p_mode='fast', channels=0, norm_p=0, abs_p=0, mean_p=0):
        super(ChannelMod, self).__init__()
        self.p_mode = p_mode
        self.channels = channels
        self.norm_p = norm_p
        self.abs_p = abs_p
        self.mean_p = mean_p
        self.enabled = False
        if self.norm_p > 0 and self.p_mode == 'slow':
            self.zero_weak_norm = ChannelMask('zero_weak', self.channels, 'norm', channel_percent=self.norm_p)
        if self.abs_p > 0 and self.p_mode == 'slow':
            self.zero_weak_abs = ChannelMask('zero_weak', self.channels, 'sum', channel_percent=self.abs_p)
        if self.mean_p > 0 and self.p_mode == 'slow':
            self.zero_weak_mean = ChannelMask('zero_weak', self.channels, 'mean', channel_percent=self.mean_p)
        if self.norm_p > 0 or self.abs_p > 0 or self.mean_p > 0:
            self.enabled = True

    def forward(self, input):
        if self.norm_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.norm_p, mode='abs-norm')
        if self.abs_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.abs_p, mode='abs-sum')
        if self.mean_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.mean_p, mode='abs-mean')
        if self.norm_p > 0 and self.p_mode == 'slow':
            self.zero_weak_norm.capture(input.clone())
            input = self.zero_weak_norm(input)
        if self.abs_p > 0 and self.p_mode == 'slow':
            self.zero_weak_abs.capture(input.clone())
            input = self.zero_weak_abs(input)
        if self.mean_p > 0 and self.p_mode == 'slow':
            self.zero_weak_mean.capture(input.clone())
            input = self.zero_weak_mean(input)
        return input


class GaussianBlur(nn.Module):

    def __init__(self, k_size, sigma):
        super(GaussianBlur, self).__init__()
        self.k_size = k_size
        self.sigma = sigma

    def capture(self, input):
        if input.dim() == 4:
            d_val = 2
            self.groups = input.size(1)
        elif input.dim() == 2:
            d_val = 1
            self.groups = input.size(0)
        self.k_size, self.sigma = [self.k_size] * d_val, [self.sigma] * d_val
        kernel = 1
        meshgrid_tensor = torch.meshgrid([torch.arange(size, dtype=torch.float32, device=input.get_device()) for size in self.k_size])
        for size, std, mgrid in zip(self.k_size, self.sigma, meshgrid_tensor):
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - (size - 1) / 2) / std) ** 2 / 2)
        kernel = (kernel / torch.sum(kernel)).view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.groups, *([1] * (kernel.dim() - 1)))
        self.register_buffer('weight', kernel)
        if d_val == 2:
            self.conv = torch.nn.functional.conv2d
        elif d_val == 1:
            self.conv = torch.nn.functional.conv1d

    def forward(self, input, pad_mode='reflect'):
        d_val = input.dim()
        if input.dim() > 2:
            input = torch.nn.functional.pad(input, (2, 2, 2, 2), mode=pad_mode)
        else:
            input = input.view(1, 1, input.size(1))
        input = self.conv(input, weight=self.weight, groups=self.groups)
        if d_val == 2:
            p1d = nn.ConstantPad1d(2, 0)
            input = p1d(input)
            input = input.view(1, input.size(2))
        return input


class GaussianBlurLP(GaussianBlur):

    def __init__(self, input, k_size=5, sigma=0):
        super(GaussianBlur, self).__init__()
        self.gauss_blur = GaussianBlur(k_size, sigma)
        self.gauss_blur.capture(input)

    def forward(self, input):
        return self.gauss_blur(input)


class GaussianBlurLayer(nn.Module):

    def __init__(self, k_size=5, sigma=0):
        super(GaussianBlurLayer, self).__init__()
        self.blur = GaussianBlur(k_size, sigma)
        self.mode = 'None'

    def forward(self, input):
        if self.mode == 'loss':
            input = self.blur(input)
        if self.mode == 'None':
            self.mode = 'capture'
        if self.mode == 'capture':
            self.blur.capture(input.clone())
            self.mode = 'loss'
        return input


class LaplacianPyramid(nn.Module):

    def __init__(self, input, scale=4, sigma=1):
        super(LaplacianPyramid, self).__init__()
        if len(sigma) == 1:
            sigma = float(sigma[0]), float(sigma[0]) * scale
        else:
            sigma = [float(s) for s in sigma]
        self.gauss_blur = GaussianBlurLP(input, 5, sigma[0])
        self.gauss_blur_hi = GaussianBlurLP(input, 5, sigma[1])
        self.scale = scale

    def split_lap(self, input):
        g = self.gauss_blur(input)
        gt = self.gauss_blur_hi(input)
        return g, input - gt

    def pyramid_list(self, input):
        pyramid_levels = []
        for i in range(self.scale):
            input, hi = self.split_lap(input)
            pyramid_levels.append(hi)
        pyramid_levels.append(input)
        return pyramid_levels[::-1]

    def lap_merge(self, pyramid_levels):
        b = torch.zeros_like(pyramid_levels[0])
        for p in pyramid_levels:
            b = b + p
        return b

    def forward(self, input):
        return self.lap_merge(self.pyramid_list(input))


class RankChannels(torch.nn.Module):

    def __init__(self, channels=1, channel_mode='strong'):
        super(RankChannels, self).__init__()
        self.channels = channels
        self.channel_mode = channel_mode

    def sort_channels(self, input):
        channel_list = []
        for i in range(input.size(1)):
            channel_list.append(torch.mean(input.clone().squeeze(0).narrow(0, i, 1)).item())
        return sorted((c, v) for v, c in enumerate(channel_list))

    def get_middle(self, sequence):
        num = self.channels[0]
        m = (len(sequence) - 1) // 2 - num // 2
        return sequence[m:m + num]

    def remove_channels(self, cl):
        return [c for c in cl if c[1] not in self.channels]

    def rank_channel_list(self, input):
        top_channels = self.channels[0]
        channel_list = self.sort_channels(input)
        if 'strong' in self.channel_mode:
            channel_list.reverse()
        elif 'avg' in self.channel_mode:
            channel_list = self.get_middle(channel_list)
        elif 'ignore' in self.channel_mode:
            channel_list = self.remove_channels(channel_list)
            top_channels = len(channel_list)
        channels = []
        for i in range(top_channels):
            channels.append(channel_list[i][1])
        return channels

    def forward(self, input):
        return self.rank_channel_list(input)


class FFTTensor(nn.Module):

    def __init__(self, r=1, bl=25):
        super(FFTTensor, self).__init__()
        self.r = r
        self.bl = bl

    def block_low(self, input):
        if input.dim() == 5:
            hh, hw = int(input.size(2) / 2), int(input.size(3) / 2)
            input[:, :, hh - self.bl:hh + self.bl + 1, hw - self.bl:hw + self.bl + 1, :] = self.r
        elif input.dim() == 3:
            m = (input.size(1) - 1) // 2 - self.bl // 2
            input[:, m:m + self.bl, :] = self.r
        return input

    def fft_image(self, input, s=0):
        s_dim = 3 if input.dim() == 4 else 1
        s_dim = s_dim if s == 0 else s
        input = torch.rfft(input, signal_ndim=s_dim, onesided=False)
        real, imaginary = torch.unbind(input, -1)
        for r_dim in range(1, len(real.size())):
            n_shift = real.size(r_dim) // 2
            if real.size(r_dim) % 2 != 0:
                n_shift += 1
            real = torch.roll(real, n_shift, dims=r_dim)
            imaginary = torch.roll(imaginary, n_shift, dims=r_dim)
        return torch.stack((real, imaginary), -1)

    def ifft_image(self, input, s=0):
        s_dim = 3 if input.dim() == 5 else 1
        s_dim = s_dim if s == 0 else s
        real, imaginary = torch.unbind(input, -1)
        for r_dim in range(len(real.size()) - 1, 0, -1):
            real = torch.roll(real, real.size(r_dim) // 2, dims=r_dim)
            imaginary = torch.roll(imaginary, imaginary.size(r_dim) // 2, dims=r_dim)
        return torch.irfft(torch.stack((real, imaginary), -1), signal_ndim=s_dim, onesided=False)

    def forward(self, input):
        input = self.block_low(self.fft_image(input))
        return torch.abs(self.ifft_image(input))


class Jitter(torch.nn.Module):

    def __init__(self, jitter_val):
        super(Jitter, self).__init__()
        self.jitter_val = jitter_val

    def roll_tensor(self, input):
        h_shift = random.randint(-self.jitter_val, self.jitter_val)
        w_shift = random.randint(-self.jitter_val, self.jitter_val)
        return torch.roll(torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3)

    def forward(self, input):
        return self.roll_tensor(input)


class RandomTransform(torch.nn.Module):

    def __init__(self, t_val):
        super(RandomTransform, self).__init__()
        self.rotate, self.flip = False, False
        if t_val == 'all' or t_val == 'rotate':
            self.rotate = True
        if t_val == 'all' or t_val == 'flip':
            self.flip = True

    def rotate_tensor(self, input):
        if self.rotate:
            k_val = random.randint(0, 3)
            input = torch.rot90(input, k_val, [2, 3])
        return input

    def flip_tensor(self, input):
        if self.flip:
            flip_tensor = bool(random.randint(0, 1))
            if flip_tensor:
                input = input.flip([2, 3])
        return input

    def forward(self, input):
        return self.flip_tensor(self.rotate_tensor(input))


class Classify(nn.Module):

    def __init__(self, labels, k=1):
        super(Classify, self).__init__()
        self.labels = [str(n) for n in labels]
        self.k = k

    def forward(self, input):
        channel_ids = torch.topk(input, self.k).indices
        channel_ids = [n.item() for n in channel_ids[0]]
        label_names = ''
        for i in channel_ids:
            if label_names != '':
                label_names += ', ' + self.labels[i]
            else:
                label_names += self.labels[i]
        None


class ModelPlus(nn.Module):

    def __init__(self, input_net, net):
        super(ModelPlus, self).__init__()
        self.input_net = input_net
        self.net = net

    def forward(self, input):
        return self.net(self.input_net(input))


class AdditionLayer(nn.Module):

    def forward(self, input, input2):
        return input + input2


class MaxPool2dLayer(nn.Module):

    def forward(self, input, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False):
        return F.max_pool2d(input, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


class PadLayer(nn.Module):

    def forward(self, input, padding=(1, 1, 1, 1), value=None):
        if value == None:
            return F.pad(input, padding)
        else:
            return F.pad(input, padding, value=value)


class ReluLayer(nn.Module):

    def forward(self, input):
        return F.relu(input)


class SoftMaxLayer(nn.Module):

    def forward(self, input, dim=1):
        return F.softmax(input, dim=dim)


class DropoutLayer(nn.Module):

    def forward(self, input, p=0.4000000059604645, training=False, inplace=True):
        return F.dropout(input=input, p=p, training=training, inplace=inplace)


class CatLayer(nn.Module):

    def forward(self, input_list, dim=1):
        return torch.cat(input_list, dim)


class LocalResponseNormLayer(nn.Module):

    def forward(self, input, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0):
        return F.local_response_norm(input, size=size, alpha=alpha, beta=beta, k=k)


class AVGPoolLayer(nn.Module):

    def forward(self, input, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False):
        return F.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)


class DreamLossMode(torch.nn.Module):

    def __init__(self, loss_mode, use_fft, r):
        super(DreamLossMode, self).__init__()
        self.get_mode(loss_mode)
        self.use_fft = use_fft[0]
        self.fft_tensor = dream_utils.FFTTensor(r, use_fft[1])

    def get_mode(self, loss_mode):
        self.loss_mode_string = loss_mode
        if loss_mode.lower() == 'norm':
            self.get_loss = self.norm_loss
        elif loss_mode.lower() == 'mean':
            self.get_loss = self.mean_loss
        elif loss_mode.lower() == 'l2':
            self.get_loss = self.l2_loss
        elif loss_mode.lower() == 'mse':
            self.crit = torch.nn.MSELoss()
            self.get_loss = self.crit_loss
        elif loss_mode.lower() == 'bce':
            self.crit = torch.nn.BCEWithLogitsLoss()
            self.get_loss = self.crit_loss
        elif loss_mode.lower() == 'abs_mean':
            self.get_loss = self.abs_mean
        elif loss_mode.lower() == 'abs_l2':
            self.get_loss = self.abs_l2_loss

    def norm_loss(self, input):
        return input.norm()

    def mean_loss(self, input):
        return input.mean()

    def l2_loss(self, input):
        return input.pow(2).sum().sqrt()

    def abs_mean(self, input):
        return input.abs().mean()

    def abs_l2_loss(self, input):
        return input.abs().pow(2).sum().sqrt()

    def crit_loss(self, input, target):
        return self.crit(input, target)

    def forward(self, input):
        if self.use_fft:
            input = self.fft_tensor(input)
        if self.loss_mode_string != 'bce' and self.loss_mode_string != 'mse':
            loss = self.get_loss(input)
        else:
            target = torch.zeros_like(input.detach())
            loss = self.crit_loss(input, target)
        return loss


class DeepDream(torch.nn.Module):

    def __init__(self, loss_mode, channels='-1', channel_mode='strong', channel_capture='once', scale=4, sigma=1, use_fft=(True, 25), r=1, p_mode='fast', norm_p=0, abs_p=0, mean_p=0):
        super(DeepDream, self).__init__()
        self.get_loss = DreamLossMode(loss_mode, use_fft, r)
        self.channels = [int(c) for c in channels.split(',')]
        self.channel_mode = channel_mode
        self.get_channels = dream_utils.RankChannels(self.channels, self.channel_mode)
        self.lap_scale = scale
        self.sigma = sigma.split(',')
        self.channel_capture = channel_capture
        self.zero_weak = ChannelMod(p_mode, self.channels[0], norm_p, abs_p, mean_p)

    def capture(self, input, lp=True):
        if -1 not in self.channels and 'all' not in self.channel_mode:
            self.channels = self.get_channels(input)
        elif self.channel_mode == 'all' and -1 not in self.channels:
            self.channels = self.channels
        if self.lap_scale > 0 and lp == True:
            self.lap_pyramid = dream_utils.LaplacianPyramid(input.clone(), self.lap_scale, self.sigma)

    def get_channel_loss(self, input):
        loss = 0
        if 'once' not in self.channel_capture:
            self.capture(input, False)
        for c in self.channels:
            if input.dim() > 0:
                if int(c) < input.size(1):
                    loss += self.get_loss(input[:, int(c)])
                else:
                    loss += self.get_loss(input)
        return loss

    def forward(self, input):
        if self.lap_scale > 0:
            input = self.lap_pyramid(input)
        if self.zero_weak.enabled:
            input = self.zero_weak(input)
        if -1 in self.channels:
            loss = self.get_loss(input)
        else:
            loss = self.get_channel_loss(input)
        return loss


class DreamLoss(torch.nn.Module):

    def __init__(self, loss_mode, strength, channels, channel_mode='all', **kwargs):
        super(DreamLoss, self).__init__()
        self.dream = DeepDream(loss_mode, channels, channel_mode, **kwargs)
        self.strength = strength
        self.mode = 'None'

    def forward(self, input):
        if self.mode == 'loss':
            self.loss = self.dream(input.clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = input.size()
            self.dream.capture(input.clone())
        return input


class DreamLossHook(DreamLoss):

    def forward(self, module, input, output):
        if self.mode == 'loss':
            self.loss = self.dream(output.clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = output.size()
            self.dream.capture(output.clone())


class DreamLossPreHook(DreamLoss):

    def forward(self, module, output):
        if self.mode == 'loss':
            self.loss = self.dream(output[0].clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = output[0].size()
            self.dream.capture(output[0].clone())


class L2Regularizer(nn.Module):

    def __init__(self, strength):
        super(L2Regularizer, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.loss = self.strength * (input.clone().norm(3) / 2)
        return input


class TVLoss(nn.Module):

    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


class BVLC_GOOGLENET(nn.Module):

    def __init__(self):
        super(BVLC_GOOGLENET, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2_3x3_reduce = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_1x1 = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_1x1 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss1_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss1_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss1_classifier_1 = nn.Linear(in_features=1024, out_features=1000, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_1x1 = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_1x1 = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss2_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_5a_1x1 = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_classifier_1 = nn.Linear(in_features=1024, out_features=1000, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_1x1 = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)

    def add_layers(self):
        self.conv1_relu_7x7 = helper_layers.ReluLayer()
        self.conv2_relu_3x3_reduce = helper_layers.ReluLayer()
        self.conv2_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5 = helper_layers.ReluLayer()
        self.pool1_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.pool2_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_3a_pool = helper_layers.MaxPool2dLayer()
        self.inception_3b_pool = helper_layers.MaxPool2dLayer()
        self.pool3_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_4a_pool = helper_layers.MaxPool2dLayer()
        self.inception_4b_pool = helper_layers.MaxPool2dLayer()
        self.inception_4c_pool = helper_layers.MaxPool2dLayer()
        self.inception_4d_pool = helper_layers.MaxPool2dLayer()
        self.inception_4e_pool = helper_layers.MaxPool2dLayer()
        self.pool4_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_5a_pool = helper_layers.MaxPool2dLayer()
        self.inception_5b_pool = helper_layers.MaxPool2dLayer()
        self.inception_3a_output = helper_layers.CatLayer()
        self.inception_3b_output = helper_layers.CatLayer()
        self.inception_4a_output = helper_layers.CatLayer()
        self.inception_4b_output = helper_layers.CatLayer()
        self.inception_4c_output = helper_layers.CatLayer()
        self.inception_4d_output = helper_layers.CatLayer()
        self.inception_4e_output = helper_layers.CatLayer()
        self.inception_5a_output = helper_layers.CatLayer()
        self.inception_5b_output = helper_layers.CatLayer()
        self.pool5_drop_7x7_s1 = helper_layers.DropoutLayer()

    def forward(self, x):
        conv1_7x7_s2_pad = F.pad(x, (3, 3, 3, 3))
        conv1_7x7_s2 = self.conv1_7x7_s2(conv1_7x7_s2_pad)
        conv1_relu_7x7 = self.conv1_relu_7x7(conv1_7x7_s2)
        pool1_3x3_s2_pad = F.pad(conv1_relu_7x7, (0, 1, 0, 1), value=float('-inf'))
        pool1_3x3_s2 = self.pool1_3x3_s2(pool1_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        pool1_norm1 = F.local_response_norm(pool1_3x3_s2, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_3x3_reduce = self.conv2_3x3_reduce(pool1_norm1)
        conv2_relu_3x3_reduce = self.conv2_relu_3x3_reduce(conv2_3x3_reduce)
        conv2_3x3_pad = F.pad(conv2_relu_3x3_reduce, (1, 1, 1, 1))
        conv2_3x3 = self.conv2_3x3(conv2_3x3_pad)
        conv2_relu_3x3 = self.conv2_relu_3x3(conv2_3x3)
        conv2_norm2 = F.local_response_norm(conv2_relu_3x3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_3x3_s2_pad = F.pad(conv2_norm2, (0, 1, 0, 1), value=float('-inf'))
        pool2_3x3_s2 = self.pool2_3x3_s2(pool2_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_pool_pad = F.pad(pool2_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool = self.inception_3a_pool(inception_3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3a_1x1 = self.inception_3a_1x1(pool2_3x3_s2)
        inception_3a_5x5_reduce = self.inception_3a_5x5_reduce(pool2_3x3_s2)
        inception_3a_3x3_reduce = self.inception_3a_3x3_reduce(pool2_3x3_s2)
        inception_3a_pool_proj = self.inception_3a_pool_proj(inception_3a_pool)
        inception_3a_relu_1x1 = self.inception_3a_relu_1x1(inception_3a_1x1)
        inception_3a_relu_5x5_reduce = self.inception_3a_relu_5x5_reduce(inception_3a_5x5_reduce)
        inception_3a_relu_3x3_reduce = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce)
        inception_3a_relu_pool_proj = self.inception_3a_relu_pool_proj(inception_3a_pool_proj)
        inception_3a_5x5_pad = F.pad(inception_3a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3a_5x5 = self.inception_3a_5x5(inception_3a_5x5_pad)
        inception_3a_3x3_pad = F.pad(inception_3a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3a_3x3 = self.inception_3a_3x3(inception_3a_3x3_pad)
        inception_3a_relu_5x5 = self.inception_3a_relu_5x5(inception_3a_5x5)
        inception_3a_relu_3x3 = self.inception_3a_relu_3x3(inception_3a_3x3)
        inception_3a_output = self.inception_3a_output((inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj), 1)
        inception_3b_3x3_reduce = self.inception_3b_3x3_reduce(inception_3a_output)
        inception_3b_pool_pad = F.pad(inception_3a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool = self.inception_3b_pool(inception_3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3b_1x1 = self.inception_3b_1x1(inception_3a_output)
        inception_3b_5x5_reduce = self.inception_3b_5x5_reduce(inception_3a_output)
        inception_3b_relu_3x3_reduce = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce)
        inception_3b_pool_proj = self.inception_3b_pool_proj(inception_3b_pool)
        inception_3b_relu_1x1 = self.inception_3b_relu_1x1(inception_3b_1x1)
        inception_3b_relu_5x5_reduce = self.inception_3b_relu_5x5_reduce(inception_3b_5x5_reduce)
        inception_3b_3x3_pad = F.pad(inception_3b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3b_3x3 = self.inception_3b_3x3(inception_3b_3x3_pad)
        inception_3b_relu_pool_proj = self.inception_3b_relu_pool_proj(inception_3b_pool_proj)
        inception_3b_5x5_pad = F.pad(inception_3b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3b_5x5 = self.inception_3b_5x5(inception_3b_5x5_pad)
        inception_3b_relu_3x3 = self.inception_3b_relu_3x3(inception_3b_3x3)
        inception_3b_relu_5x5 = self.inception_3b_relu_5x5(inception_3b_5x5)
        inception_3b_output = self.inception_3b_output((inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj), 1)
        pool3_3x3_s2_pad = F.pad(inception_3b_output, (0, 1, 0, 1), value=float('-inf'))
        pool3_3x3_s2 = self.pool3_3x3_s2(pool3_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_1x1 = self.inception_4a_1x1(pool3_3x3_s2)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(pool3_3x3_s2)
        inception_4a_5x5_reduce = self.inception_4a_5x5_reduce(pool3_3x3_s2)
        inception_4a_pool_pad = F.pad(pool3_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool = self.inception_4a_pool(inception_4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4a_relu_1x1 = self.inception_4a_relu_1x1(inception_4a_1x1)
        inception_4a_relu_3x3_reduce = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce)
        inception_4a_relu_5x5_reduce = self.inception_4a_relu_5x5_reduce(inception_4a_5x5_reduce)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        inception_4a_3x3_pad = F.pad(inception_4a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_pad)
        inception_4a_5x5_pad = F.pad(inception_4a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4a_5x5 = self.inception_4a_5x5(inception_4a_5x5_pad)
        inception_4a_relu_pool_proj = self.inception_4a_relu_pool_proj(inception_4a_pool_proj)
        inception_4a_relu_3x3 = self.inception_4a_relu_3x3(inception_4a_3x3)
        inception_4a_relu_5x5 = self.inception_4a_relu_5x5(inception_4a_5x5)
        inception_4a_output = self.inception_4a_output((inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj), 1)
        inception_4b_pool_pad = F.pad(inception_4a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool = self.inception_4b_pool(inception_4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4b_5x5_reduce = self.inception_4b_5x5_reduce(inception_4a_output)
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        inception_4b_relu_5x5_reduce = self.inception_4b_relu_5x5_reduce(inception_4b_5x5_reduce)
        inception_4b_relu_1x1 = self.inception_4b_relu_1x1(inception_4b_1x1)
        inception_4b_relu_3x3_reduce = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce)
        inception_4b_relu_pool_proj = self.inception_4b_relu_pool_proj(inception_4b_pool_proj)
        inception_4b_5x5_pad = F.pad(inception_4b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4b_5x5 = self.inception_4b_5x5(inception_4b_5x5_pad)
        inception_4b_3x3_pad = F.pad(inception_4b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_pad)
        inception_4b_relu_5x5 = self.inception_4b_relu_5x5(inception_4b_5x5)
        inception_4b_relu_3x3 = self.inception_4b_relu_3x3(inception_4b_3x3)
        inception_4b_output = self.inception_4b_output((inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj), 1)
        inception_4c_5x5_reduce = self.inception_4c_5x5_reduce(inception_4b_output)
        inception_4c_pool_pad = F.pad(inception_4b_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool = self.inception_4c_pool(inception_4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_relu_5x5_reduce = self.inception_4c_relu_5x5_reduce(inception_4c_5x5_reduce)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_relu_1x1 = self.inception_4c_relu_1x1(inception_4c_1x1)
        inception_4c_relu_3x3_reduce = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce)
        inception_4c_5x5_pad = F.pad(inception_4c_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4c_5x5 = self.inception_4c_5x5(inception_4c_5x5_pad)
        inception_4c_relu_pool_proj = self.inception_4c_relu_pool_proj(inception_4c_pool_proj)
        inception_4c_3x3_pad = F.pad(inception_4c_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_pad)
        inception_4c_relu_5x5 = self.inception_4c_relu_5x5(inception_4c_5x5)
        inception_4c_relu_3x3 = self.inception_4c_relu_3x3(inception_4c_3x3)
        inception_4c_output = self.inception_4c_output((inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj), 1)
        inception_4d_pool_pad = F.pad(inception_4c_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool = self.inception_4d_pool(inception_4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_5x5_reduce = self.inception_4d_5x5_reduce(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_relu_3x3_reduce = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce)
        inception_4d_relu_1x1 = self.inception_4d_relu_1x1(inception_4d_1x1)
        inception_4d_relu_5x5_reduce = self.inception_4d_relu_5x5_reduce(inception_4d_5x5_reduce)
        inception_4d_relu_pool_proj = self.inception_4d_relu_pool_proj(inception_4d_pool_proj)
        inception_4d_3x3_pad = F.pad(inception_4d_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_pad)
        inception_4d_5x5_pad = F.pad(inception_4d_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4d_5x5 = self.inception_4d_5x5(inception_4d_5x5_pad)
        inception_4d_relu_3x3 = self.inception_4d_relu_3x3(inception_4d_3x3)
        inception_4d_relu_5x5 = self.inception_4d_relu_5x5(inception_4d_5x5)
        inception_4d_output = self.inception_4d_output((inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj), 1)
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_5x5_reduce = self.inception_4e_5x5_reduce(inception_4d_output)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_pool_pad = F.pad(inception_4d_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool = self.inception_4e_pool(inception_4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4e_relu_1x1 = self.inception_4e_relu_1x1(inception_4e_1x1)
        inception_4e_relu_5x5_reduce = self.inception_4e_relu_5x5_reduce(inception_4e_5x5_reduce)
        inception_4e_relu_3x3_reduce = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce)
        inception_4e_pool_proj = self.inception_4e_pool_proj(inception_4e_pool)
        inception_4e_5x5_pad = F.pad(inception_4e_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4e_5x5 = self.inception_4e_5x5(inception_4e_5x5_pad)
        inception_4e_3x3_pad = F.pad(inception_4e_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4e_3x3 = self.inception_4e_3x3(inception_4e_3x3_pad)
        inception_4e_relu_pool_proj = self.inception_4e_relu_pool_proj(inception_4e_pool_proj)
        inception_4e_relu_5x5 = self.inception_4e_relu_5x5(inception_4e_5x5)
        inception_4e_relu_3x3 = self.inception_4e_relu_3x3(inception_4e_3x3)
        inception_4e_output = self.inception_4e_output((inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj), 1)
        pool4_3x3_s2_pad = F.pad(inception_4e_output, (0, 1, 0, 1), value=float('-inf'))
        pool4_3x3_s2 = self.pool4_3x3_s2(pool4_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_5a_1x1 = self.inception_5a_1x1(pool4_3x3_s2)
        inception_5a_5x5_reduce = self.inception_5a_5x5_reduce(pool4_3x3_s2)
        inception_5a_pool_pad = F.pad(pool4_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool = self.inception_5a_pool(inception_5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5a_3x3_reduce = self.inception_5a_3x3_reduce(pool4_3x3_s2)
        inception_5a_relu_1x1 = self.inception_5a_relu_1x1(inception_5a_1x1)
        inception_5a_relu_5x5_reduce = self.inception_5a_relu_5x5_reduce(inception_5a_5x5_reduce)
        inception_5a_pool_proj = self.inception_5a_pool_proj(inception_5a_pool)
        inception_5a_relu_3x3_reduce = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce)
        inception_5a_5x5_pad = F.pad(inception_5a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5a_5x5 = self.inception_5a_5x5(inception_5a_5x5_pad)
        inception_5a_relu_pool_proj = self.inception_5a_relu_pool_proj(inception_5a_pool_proj)
        inception_5a_3x3_pad = F.pad(inception_5a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5a_3x3 = self.inception_5a_3x3(inception_5a_3x3_pad)
        inception_5a_relu_5x5 = self.inception_5a_relu_5x5(inception_5a_5x5)
        inception_5a_relu_3x3 = self.inception_5a_relu_3x3(inception_5a_3x3)
        inception_5a_output = self.inception_5a_output((inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj), 1)
        inception_5b_3x3_reduce = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_pool_pad = F.pad(inception_5a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool = self.inception_5b_pool(inception_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5b_5x5_reduce = self.inception_5b_5x5_reduce(inception_5a_output)
        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)
        inception_5b_relu_3x3_reduce = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce)
        inception_5b_pool_proj = self.inception_5b_pool_proj(inception_5b_pool)
        inception_5b_relu_5x5_reduce = self.inception_5b_relu_5x5_reduce(inception_5b_5x5_reduce)
        inception_5b_relu_1x1 = self.inception_5b_relu_1x1(inception_5b_1x1)
        inception_5b_3x3_pad = F.pad(inception_5b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5b_3x3 = self.inception_5b_3x3(inception_5b_3x3_pad)
        inception_5b_relu_pool_proj = self.inception_5b_relu_pool_proj(inception_5b_pool_proj)
        inception_5b_5x5_pad = F.pad(inception_5b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5b_5x5 = self.inception_5b_5x5(inception_5b_5x5_pad)
        inception_5b_relu_3x3 = self.inception_5b_relu_3x3(inception_5b_3x3)
        inception_5b_relu_5x5 = self.inception_5b_relu_5x5(inception_5b_5x5)
        inception_5b_output = self.inception_5b_output((inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj), 1)
        pool5_7x7_s1 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        pool5_drop_7x7_s1 = self.pool5_drop_7x7_s1(input=pool5_7x7_s1, p=0.4000000059604645, training=self.training, inplace=True)
        return pool5_drop_7x7_s1


class GOOGLENET_CARS(nn.Module):

    def __init__(self):
        super(GOOGLENET_CARS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_1x1 = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_1x1 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss1_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.loss1_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_4c_1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss1_classifier_model_1 = nn.Linear(in_features=1024, out_features=431, bias=True)
        self.inception_4d_1x1 = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_1x1 = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.loss2_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_1x1 = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_classifier_model_1 = nn.Linear(in_features=1024, out_features=431, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_1x1 = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)

    def add_layers(self):
        self.relu1 = helper_layers.ReluLayer()
        self.relu_conv2_1x1 = helper_layers.ReluLayer()
        self.relu2_3x3 = helper_layers.ReluLayer()
        self.relu_inception_3a_5x5_reduce = helper_layers.ReluLayer()
        self.reulu_inception_3a_3x3_reduce = helper_layers.ReluLayer()
        self.relu_inception_3a_1x1 = helper_layers.ReluLayer()
        self.relu_inception_3a_pool_proj = helper_layers.ReluLayer()
        self.relu_inception_3a_5x5 = helper_layers.ReluLayer()
        self.relu_inception_3a_3x3 = helper_layers.ReluLayer()
        self.relu_inception_3b_1x1 = helper_layers.ReluLayer()
        self.relu_inception_3b_3x3_reduce = helper_layers.ReluLayer()
        self.relu_inception_3b_5x5_reduce = helper_layers.ReluLayer()
        self.relu_inception_3b_pool_proj = helper_layers.ReluLayer()
        self.relu_inception_3b_3x3 = helper_layers.ReluLayer()
        self.relu_inception_3b_5x5 = helper_layers.ReluLayer()
        self.relu_inception_4a_3x3_reduce = helper_layers.ReluLayer()
        self.relu_inception_4a_5x5_reduce = helper_layers.ReluLayer()
        self.relu_inception_4a_1x1 = helper_layers.ReluLayer()
        self.relu_inception_4a_pool_proj = helper_layers.ReluLayer()
        self.relu_inception_4a_3x3 = helper_layers.ReluLayer()
        self.relu_inception_4a_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4c_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5 = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5 = helper_layers.ReluLayer()
        self.pool1 = helper_layers.MaxPool2dLayer()
        self.pool2 = helper_layers.MaxPool2dLayer()
        self.inception_3a_pool = helper_layers.MaxPool2dLayer()
        self.inception_3b_pool = helper_layers.MaxPool2dLayer()
        self.pool3 = helper_layers.MaxPool2dLayer()
        self.inception_4a_pool = helper_layers.MaxPool2dLayer()
        self.inception_4b_pool = helper_layers.MaxPool2dLayer()
        self.inception_4c_pool = helper_layers.MaxPool2dLayer()
        self.inception_4d_pool = helper_layers.MaxPool2dLayer()
        self.inception_4e_pool = helper_layers.MaxPool2dLayer()
        self.pool4 = helper_layers.MaxPool2dLayer()
        self.inception_5a_pool = helper_layers.MaxPool2dLayer()
        self.inception_5b_pool = helper_layers.MaxPool2dLayer()
        self.inception_3a_output = helper_layers.CatLayer()
        self.inception_3b_output = helper_layers.CatLayer()
        self.inception_4a_output = helper_layers.CatLayer()
        self.inception_4b_output = helper_layers.CatLayer()
        self.inception_4c_output = helper_layers.CatLayer()
        self.inception_4d_output = helper_layers.CatLayer()
        self.inception_4e_output = helper_layers.CatLayer()
        self.inception_5a_output = helper_layers.CatLayer()
        self.inception_5b_output = helper_layers.CatLayer()
        self.pool5_drop = helper_layers.DropoutLayer()

    def forward(self, x):
        conv1_pad = F.pad(x, (3, 3, 3, 3))
        conv1 = self.conv1(conv1_pad)
        relu1 = self.relu1(conv1)
        pool1_pad = F.pad(relu1, (0, 1, 0, 1), value=float('-inf'))
        pool1 = self.pool1(pool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        norm1 = F.local_response_norm(pool1, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_1x1 = self.conv2_1x1(norm1)
        relu_conv2_1x1 = self.relu_conv2_1x1(conv2_1x1)
        conv2_3x3_pad = F.pad(relu_conv2_1x1, (1, 1, 1, 1))
        conv2_3x3 = self.conv2_3x3(conv2_3x3_pad)
        relu2_3x3 = self.relu2_3x3(conv2_3x3)
        norm2 = F.local_response_norm(relu2_3x3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_pad = F.pad(norm2, (0, 1, 0, 1), value=float('-inf'))
        pool2 = self.pool2(pool2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_5x5_reduce = self.inception_3a_5x5_reduce(pool2)
        inception_3a_3x3_reduce = self.inception_3a_3x3_reduce(pool2)
        inception_3a_1x1 = self.inception_3a_1x1(pool2)
        inception_3a_pool_pad = F.pad(pool2, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool = self.inception_3a_pool(inception_3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        relu_inception_3a_5x5_reduce = self.relu_inception_3a_5x5_reduce(inception_3a_5x5_reduce)
        reulu_inception_3a_3x3_reduce = self.reulu_inception_3a_3x3_reduce(inception_3a_3x3_reduce)
        relu_inception_3a_1x1 = self.relu_inception_3a_1x1(inception_3a_1x1)
        inception_3a_pool_proj = self.inception_3a_pool_proj(inception_3a_pool)
        inception_3a_5x5_pad = F.pad(relu_inception_3a_5x5_reduce, (2, 2, 2, 2))
        inception_3a_5x5 = self.inception_3a_5x5(inception_3a_5x5_pad)
        inception_3a_3x3_pad = F.pad(reulu_inception_3a_3x3_reduce, (1, 1, 1, 1))
        inception_3a_3x3 = self.inception_3a_3x3(inception_3a_3x3_pad)
        relu_inception_3a_pool_proj = self.relu_inception_3a_pool_proj(inception_3a_pool_proj)
        relu_inception_3a_5x5 = self.relu_inception_3a_5x5(inception_3a_5x5)
        relu_inception_3a_3x3 = self.relu_inception_3a_3x3(inception_3a_3x3)
        inception_3a_output = self.inception_3a_output((relu_inception_3a_1x1, relu_inception_3a_3x3, relu_inception_3a_5x5, relu_inception_3a_pool_proj), 1)
        inception_3b_1x1 = self.inception_3b_1x1(inception_3a_output)
        inception_3b_3x3_reduce = self.inception_3b_3x3_reduce(inception_3a_output)
        inception_3b_5x5_reduce = self.inception_3b_5x5_reduce(inception_3a_output)
        inception_3b_pool_pad = F.pad(inception_3a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool = self.inception_3b_pool(inception_3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        relu_inception_3b_1x1 = self.relu_inception_3b_1x1(inception_3b_1x1)
        relu_inception_3b_3x3_reduce = self.relu_inception_3b_3x3_reduce(inception_3b_3x3_reduce)
        relu_inception_3b_5x5_reduce = self.relu_inception_3b_5x5_reduce(inception_3b_5x5_reduce)
        inception_3b_pool_proj = self.inception_3b_pool_proj(inception_3b_pool)
        inception_3b_3x3_pad = F.pad(relu_inception_3b_3x3_reduce, (1, 1, 1, 1))
        inception_3b_3x3 = self.inception_3b_3x3(inception_3b_3x3_pad)
        inception_3b_5x5_pad = F.pad(relu_inception_3b_5x5_reduce, (2, 2, 2, 2))
        inception_3b_5x5 = self.inception_3b_5x5(inception_3b_5x5_pad)
        relu_inception_3b_pool_proj = self.relu_inception_3b_pool_proj(inception_3b_pool_proj)
        relu_inception_3b_3x3 = self.relu_inception_3b_3x3(inception_3b_3x3)
        relu_inception_3b_5x5 = self.relu_inception_3b_5x5(inception_3b_5x5)
        inception_3b_output = self.inception_3b_output((relu_inception_3b_1x1, relu_inception_3b_3x3, relu_inception_3b_5x5, relu_inception_3b_pool_proj), 1)
        pool3_pad = F.pad(inception_3b_output, (0, 1, 0, 1), value=float('-inf'))
        pool3 = self.pool3(pool3_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_pool_pad = F.pad(pool3, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool = self.inception_4a_pool(inception_4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(pool3)
        inception_4a_5x5_reduce = self.inception_4a_5x5_reduce(pool3)
        inception_4a_1x1 = self.inception_4a_1x1(pool3)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        relu_inception_4a_3x3_reduce = self.relu_inception_4a_3x3_reduce(inception_4a_3x3_reduce)
        relu_inception_4a_5x5_reduce = self.relu_inception_4a_5x5_reduce(inception_4a_5x5_reduce)
        relu_inception_4a_1x1 = self.relu_inception_4a_1x1(inception_4a_1x1)
        relu_inception_4a_pool_proj = self.relu_inception_4a_pool_proj(inception_4a_pool_proj)
        inception_4a_3x3_pad = F.pad(relu_inception_4a_3x3_reduce, (1, 1, 1, 1))
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_pad)
        inception_4a_5x5_pad = F.pad(relu_inception_4a_5x5_reduce, (2, 2, 2, 2))
        inception_4a_5x5 = self.inception_4a_5x5(inception_4a_5x5_pad)
        relu_inception_4a_3x3 = self.relu_inception_4a_3x3(inception_4a_3x3)
        relu_inception_4a_5x5 = self.relu_inception_4a_5x5(inception_4a_5x5)
        inception_4a_output = self.inception_4a_output((relu_inception_4a_1x1, relu_inception_4a_3x3, relu_inception_4a_5x5, relu_inception_4a_pool_proj), 1)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_pool_pad = F.pad(inception_4a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool = self.inception_4b_pool(inception_4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4b_5x5_reduce = self.inception_4b_5x5_reduce(inception_4a_output)
        inception_4b_relu_3x3_reduce = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce)
        inception_4b_relu_1x1 = self.inception_4b_relu_1x1(inception_4b_1x1)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        inception_4b_relu_5x5_reduce = self.inception_4b_relu_5x5_reduce(inception_4b_5x5_reduce)
        inception_4b_3x3_pad = F.pad(inception_4b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_pad)
        inception_4b_relu_pool_proj = self.inception_4b_relu_pool_proj(inception_4b_pool_proj)
        inception_4b_5x5_pad = F.pad(inception_4b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4b_5x5 = self.inception_4b_5x5(inception_4b_5x5_pad)
        inception_4b_relu_3x3 = self.inception_4b_relu_3x3(inception_4b_3x3)
        inception_4b_relu_5x5 = self.inception_4b_relu_5x5(inception_4b_5x5)
        inception_4b_output = self.inception_4b_output((inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj), 1)
        inception_4c_pool_pad = F.pad(inception_4b_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool = self.inception_4c_pool(inception_4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_5x5_reduce = self.inception_4c_5x5_reduce(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_relu_1x1 = self.inception_4c_relu_1x1(inception_4c_1x1)
        inception_4c_relu_5x5_reduce = self.inception_4c_relu_5x5_reduce(inception_4c_5x5_reduce)
        inception_4c_relu_3x3_reduce = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce)
        inception_4c_relu_pool_proj = self.inception_4c_relu_pool_proj(inception_4c_pool_proj)
        inception_4c_5x5_pad = F.pad(inception_4c_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4c_5x5 = self.inception_4c_5x5(inception_4c_5x5_pad)
        inception_4c_3x3_pad = F.pad(inception_4c_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_pad)
        inception_4c_relu_5x5 = self.inception_4c_relu_5x5(inception_4c_5x5)
        inception_4c_relu_3x3 = self.inception_4c_relu_3x3(inception_4c_3x3)
        inception_4c_output = self.inception_4c_output((inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj), 1)
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_5x5_reduce = self.inception_4d_5x5_reduce(inception_4c_output)
        inception_4d_pool_pad = F.pad(inception_4c_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool = self.inception_4d_pool(inception_4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4d_relu_1x1 = self.inception_4d_relu_1x1(inception_4d_1x1)
        inception_4d_relu_3x3_reduce = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce)
        inception_4d_relu_5x5_reduce = self.inception_4d_relu_5x5_reduce(inception_4d_5x5_reduce)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_3x3_pad = F.pad(inception_4d_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_pad)
        inception_4d_5x5_pad = F.pad(inception_4d_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4d_5x5 = self.inception_4d_5x5(inception_4d_5x5_pad)
        inception_4d_relu_pool_proj = self.inception_4d_relu_pool_proj(inception_4d_pool_proj)
        inception_4d_relu_3x3 = self.inception_4d_relu_3x3(inception_4d_3x3)
        inception_4d_relu_5x5 = self.inception_4d_relu_5x5(inception_4d_5x5)
        inception_4d_output = self.inception_4d_output((inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj), 1)
        inception_4e_pool_pad = F.pad(inception_4d_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool = self.inception_4e_pool(inception_4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_5x5_reduce = self.inception_4e_5x5_reduce(inception_4d_output)
        inception_4e_pool_proj = self.inception_4e_pool_proj(inception_4e_pool)
        inception_4e_relu_1x1 = self.inception_4e_relu_1x1(inception_4e_1x1)
        inception_4e_relu_3x3_reduce = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce)
        inception_4e_relu_5x5_reduce = self.inception_4e_relu_5x5_reduce(inception_4e_5x5_reduce)
        inception_4e_relu_pool_proj = self.inception_4e_relu_pool_proj(inception_4e_pool_proj)
        inception_4e_3x3_pad = F.pad(inception_4e_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4e_3x3 = self.inception_4e_3x3(inception_4e_3x3_pad)
        inception_4e_5x5_pad = F.pad(inception_4e_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4e_5x5 = self.inception_4e_5x5(inception_4e_5x5_pad)
        inception_4e_relu_3x3 = self.inception_4e_relu_3x3(inception_4e_3x3)
        inception_4e_relu_5x5 = self.inception_4e_relu_5x5(inception_4e_5x5)
        inception_4e_output = self.inception_4e_output((inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj), 1)
        pool4_pad = F.pad(inception_4e_output, (0, 1, 0, 1), value=float('-inf'))
        pool4 = self.pool4(pool4_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_5a_pool_pad = F.pad(pool4, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool = self.inception_5a_pool(inception_5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5a_5x5_reduce = self.inception_5a_5x5_reduce(pool4)
        inception_5a_3x3_reduce = self.inception_5a_3x3_reduce(pool4)
        inception_5a_1x1 = self.inception_5a_1x1(pool4)
        inception_5a_pool_proj = self.inception_5a_pool_proj(inception_5a_pool)
        inception_5a_relu_5x5_reduce = self.inception_5a_relu_5x5_reduce(inception_5a_5x5_reduce)
        inception_5a_relu_3x3_reduce = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce)
        inception_5a_relu_1x1 = self.inception_5a_relu_1x1(inception_5a_1x1)
        inception_5a_relu_pool_proj = self.inception_5a_relu_pool_proj(inception_5a_pool_proj)
        inception_5a_5x5_pad = F.pad(inception_5a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5a_5x5 = self.inception_5a_5x5(inception_5a_5x5_pad)
        inception_5a_3x3_pad = F.pad(inception_5a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5a_3x3 = self.inception_5a_3x3(inception_5a_3x3_pad)
        inception_5a_relu_5x5 = self.inception_5a_relu_5x5(inception_5a_5x5)
        inception_5a_relu_3x3 = self.inception_5a_relu_3x3(inception_5a_3x3)
        inception_5a_output = self.inception_5a_output((inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj), 1)
        inception_5b_3x3_reduce = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)
        inception_5b_5x5_reduce = self.inception_5b_5x5_reduce(inception_5a_output)
        inception_5b_pool_pad = F.pad(inception_5a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool = self.inception_5b_pool(inception_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5b_relu_3x3_reduce = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce)
        inception_5b_relu_1x1 = self.inception_5b_relu_1x1(inception_5b_1x1)
        inception_5b_relu_5x5_reduce = self.inception_5b_relu_5x5_reduce(inception_5b_5x5_reduce)
        inception_5b_pool_proj = self.inception_5b_pool_proj(inception_5b_pool)
        inception_5b_3x3_pad = F.pad(inception_5b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5b_3x3 = self.inception_5b_3x3(inception_5b_3x3_pad)
        inception_5b_5x5_pad = F.pad(inception_5b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5b_5x5 = self.inception_5b_5x5(inception_5b_5x5_pad)
        inception_5b_relu_pool_proj = self.inception_5b_relu_pool_proj(inception_5b_pool_proj)
        inception_5b_relu_3x3 = self.inception_5b_relu_3x3(inception_5b_3x3)
        inception_5b_relu_5x5 = self.inception_5b_relu_5x5(inception_5b_5x5)
        inception_5b_output = self.inception_5b_output((inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj), 1)
        pool5 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        pool5_drop = self.pool5_drop(input=pool5, p=0.4000000059604645, training=self.training, inplace=True)
        return pool5_drop


class GoogleNet_SOS(nn.Module):

    def __init__(self):
        super(GoogleNet_SOS, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2_3x3_reduce = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_1x1 = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_1x1 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_1x1 = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_1x1 = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_1x1 = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_1x1 = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)

    def add_layers(self):
        self.conv1_relu_7x7 = helper_layers.ReluLayer()
        self.conv2_relu_3x3_reduce = helper_layers.ReluLayer()
        self.conv2_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5 = helper_layers.ReluLayer()
        self.pool1_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.pool2_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_3a_pool = helper_layers.MaxPool2dLayer()
        self.inception_3b_pool = helper_layers.MaxPool2dLayer()
        self.pool3_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_4a_pool = helper_layers.MaxPool2dLayer()
        self.inception_4b_pool = helper_layers.MaxPool2dLayer()
        self.inception_4c_pool = helper_layers.MaxPool2dLayer()
        self.inception_4d_pool = helper_layers.MaxPool2dLayer()
        self.inception_4e_pool = helper_layers.MaxPool2dLayer()
        self.pool4_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_5a_pool = helper_layers.MaxPool2dLayer()
        self.inception_5b_pool = helper_layers.MaxPool2dLayer()
        self.inception_3a_output = helper_layers.CatLayer()
        self.inception_3b_output = helper_layers.CatLayer()
        self.inception_4a_output = helper_layers.CatLayer()
        self.inception_4b_output = helper_layers.CatLayer()
        self.inception_4c_output = helper_layers.CatLayer()
        self.inception_4d_output = helper_layers.CatLayer()
        self.inception_4e_output = helper_layers.CatLayer()
        self.inception_5a_output = helper_layers.CatLayer()
        self.inception_5b_output = helper_layers.CatLayer()
        self.pool5_drop_7x7_s1 = helper_layers.DropoutLayer()

    def forward(self, x):
        conv1_7x7_s2_pad = F.pad(x, (3, 3, 3, 3))
        conv1_7x7_s2 = self.conv1_7x7_s2(conv1_7x7_s2_pad)
        conv1_relu_7x7 = self.conv1_relu_7x7(conv1_7x7_s2)
        pool1_3x3_s2_pad = F.pad(conv1_relu_7x7, (0, 1, 0, 1), value=float('-inf'))
        pool1_3x3_s2 = self.pool1_3x3_s2(pool1_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        pool1_norm1 = F.local_response_norm(pool1_3x3_s2, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_3x3_reduce = self.conv2_3x3_reduce(pool1_norm1)
        conv2_relu_3x3_reduce = self.conv2_relu_3x3_reduce(conv2_3x3_reduce)
        conv2_3x3_pad = F.pad(conv2_relu_3x3_reduce, (1, 1, 1, 1))
        conv2_3x3 = self.conv2_3x3(conv2_3x3_pad)
        conv2_relu_3x3 = self.conv2_relu_3x3(conv2_3x3)
        conv2_norm2 = F.local_response_norm(conv2_relu_3x3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_3x3_s2_pad = F.pad(conv2_norm2, (0, 1, 0, 1), value=float('-inf'))
        pool2_3x3_s2 = self.pool2_3x3_s2(pool2_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_pool_pad = F.pad(pool2_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool = self.inception_3a_pool(inception_3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3a_1x1 = self.inception_3a_1x1(pool2_3x3_s2)
        inception_3a_5x5_reduce = self.inception_3a_5x5_reduce(pool2_3x3_s2)
        inception_3a_3x3_reduce = self.inception_3a_3x3_reduce(pool2_3x3_s2)
        inception_3a_pool_proj = self.inception_3a_pool_proj(inception_3a_pool)
        inception_3a_relu_1x1 = self.inception_3a_relu_1x1(inception_3a_1x1)
        inception_3a_relu_5x5_reduce = self.inception_3a_relu_5x5_reduce(inception_3a_5x5_reduce)
        inception_3a_relu_3x3_reduce = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce)
        inception_3a_relu_pool_proj = self.inception_3a_relu_pool_proj(inception_3a_pool_proj)
        inception_3a_5x5_pad = F.pad(inception_3a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3a_5x5 = self.inception_3a_5x5(inception_3a_5x5_pad)
        inception_3a_3x3_pad = F.pad(inception_3a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3a_3x3 = self.inception_3a_3x3(inception_3a_3x3_pad)
        inception_3a_relu_5x5 = self.inception_3a_relu_5x5(inception_3a_5x5)
        inception_3a_relu_3x3 = self.inception_3a_relu_3x3(inception_3a_3x3)
        inception_3a_output = self.inception_3a_output((inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj), 1)
        inception_3b_3x3_reduce = self.inception_3b_3x3_reduce(inception_3a_output)
        inception_3b_pool_pad = F.pad(inception_3a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool = self.inception_3b_pool(inception_3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3b_1x1 = self.inception_3b_1x1(inception_3a_output)
        inception_3b_5x5_reduce = self.inception_3b_5x5_reduce(inception_3a_output)
        inception_3b_relu_3x3_reduce = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce)
        inception_3b_pool_proj = self.inception_3b_pool_proj(inception_3b_pool)
        inception_3b_relu_1x1 = self.inception_3b_relu_1x1(inception_3b_1x1)
        inception_3b_relu_5x5_reduce = self.inception_3b_relu_5x5_reduce(inception_3b_5x5_reduce)
        inception_3b_3x3_pad = F.pad(inception_3b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3b_3x3 = self.inception_3b_3x3(inception_3b_3x3_pad)
        inception_3b_relu_pool_proj = self.inception_3b_relu_pool_proj(inception_3b_pool_proj)
        inception_3b_5x5_pad = F.pad(inception_3b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3b_5x5 = self.inception_3b_5x5(inception_3b_5x5_pad)
        inception_3b_relu_3x3 = self.inception_3b_relu_3x3(inception_3b_3x3)
        inception_3b_relu_5x5 = self.inception_3b_relu_5x5(inception_3b_5x5)
        inception_3b_output = self.inception_3b_output((inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj), 1)
        pool3_3x3_s2_pad = F.pad(inception_3b_output, (0, 1, 0, 1), value=float('-inf'))
        pool3_3x3_s2 = self.pool3_3x3_s2(pool3_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_1x1 = self.inception_4a_1x1(pool3_3x3_s2)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(pool3_3x3_s2)
        inception_4a_5x5_reduce = self.inception_4a_5x5_reduce(pool3_3x3_s2)
        inception_4a_pool_pad = F.pad(pool3_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool = self.inception_4a_pool(inception_4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4a_relu_1x1 = self.inception_4a_relu_1x1(inception_4a_1x1)
        inception_4a_relu_3x3_reduce = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce)
        inception_4a_relu_5x5_reduce = self.inception_4a_relu_5x5_reduce(inception_4a_5x5_reduce)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        inception_4a_3x3_pad = F.pad(inception_4a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_pad)
        inception_4a_5x5_pad = F.pad(inception_4a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4a_5x5 = self.inception_4a_5x5(inception_4a_5x5_pad)
        inception_4a_relu_pool_proj = self.inception_4a_relu_pool_proj(inception_4a_pool_proj)
        inception_4a_relu_3x3 = self.inception_4a_relu_3x3(inception_4a_3x3)
        inception_4a_relu_5x5 = self.inception_4a_relu_5x5(inception_4a_5x5)
        inception_4a_output = self.inception_4a_output((inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj), 1)
        inception_4b_pool_pad = F.pad(inception_4a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool = self.inception_4b_pool(inception_4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4b_5x5_reduce = self.inception_4b_5x5_reduce(inception_4a_output)
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        inception_4b_relu_5x5_reduce = self.inception_4b_relu_5x5_reduce(inception_4b_5x5_reduce)
        inception_4b_relu_1x1 = self.inception_4b_relu_1x1(inception_4b_1x1)
        inception_4b_relu_3x3_reduce = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce)
        inception_4b_relu_pool_proj = self.inception_4b_relu_pool_proj(inception_4b_pool_proj)
        inception_4b_5x5_pad = F.pad(inception_4b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4b_5x5 = self.inception_4b_5x5(inception_4b_5x5_pad)
        inception_4b_3x3_pad = F.pad(inception_4b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_pad)
        inception_4b_relu_5x5 = self.inception_4b_relu_5x5(inception_4b_5x5)
        inception_4b_relu_3x3 = self.inception_4b_relu_3x3(inception_4b_3x3)
        inception_4b_output = self.inception_4b_output((inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj), 1)
        inception_4c_5x5_reduce = self.inception_4c_5x5_reduce(inception_4b_output)
        inception_4c_pool_pad = F.pad(inception_4b_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool = self.inception_4c_pool(inception_4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_relu_5x5_reduce = self.inception_4c_relu_5x5_reduce(inception_4c_5x5_reduce)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_relu_1x1 = self.inception_4c_relu_1x1(inception_4c_1x1)
        inception_4c_relu_3x3_reduce = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce)
        inception_4c_5x5_pad = F.pad(inception_4c_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4c_5x5 = self.inception_4c_5x5(inception_4c_5x5_pad)
        inception_4c_relu_pool_proj = self.inception_4c_relu_pool_proj(inception_4c_pool_proj)
        inception_4c_3x3_pad = F.pad(inception_4c_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_pad)
        inception_4c_relu_5x5 = self.inception_4c_relu_5x5(inception_4c_5x5)
        inception_4c_relu_3x3 = self.inception_4c_relu_3x3(inception_4c_3x3)
        inception_4c_output = self.inception_4c_output((inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj), 1)
        inception_4d_pool_pad = F.pad(inception_4c_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool = self.inception_4d_pool(inception_4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_5x5_reduce = self.inception_4d_5x5_reduce(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_relu_3x3_reduce = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce)
        inception_4d_relu_1x1 = self.inception_4d_relu_1x1(inception_4d_1x1)
        inception_4d_relu_5x5_reduce = self.inception_4d_relu_5x5_reduce(inception_4d_5x5_reduce)
        inception_4d_relu_pool_proj = self.inception_4d_relu_pool_proj(inception_4d_pool_proj)
        inception_4d_3x3_pad = F.pad(inception_4d_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_pad)
        inception_4d_5x5_pad = F.pad(inception_4d_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4d_5x5 = self.inception_4d_5x5(inception_4d_5x5_pad)
        inception_4d_relu_3x3 = self.inception_4d_relu_3x3(inception_4d_3x3)
        inception_4d_relu_5x5 = self.inception_4d_relu_5x5(inception_4d_5x5)
        inception_4d_output = self.inception_4d_output((inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj), 1)
        inception_4e_5x5_reduce = self.inception_4e_5x5_reduce(inception_4d_output)
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_pool_pad = F.pad(inception_4d_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool = self.inception_4e_pool(inception_4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4e_relu_5x5_reduce = self.inception_4e_relu_5x5_reduce(inception_4e_5x5_reduce)
        inception_4e_relu_1x1 = self.inception_4e_relu_1x1(inception_4e_1x1)
        inception_4e_relu_3x3_reduce = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce)
        inception_4e_pool_proj = self.inception_4e_pool_proj(inception_4e_pool)
        inception_4e_5x5_pad = F.pad(inception_4e_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4e_5x5 = self.inception_4e_5x5(inception_4e_5x5_pad)
        inception_4e_3x3_pad = F.pad(inception_4e_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4e_3x3 = self.inception_4e_3x3(inception_4e_3x3_pad)
        inception_4e_relu_pool_proj = self.inception_4e_relu_pool_proj(inception_4e_pool_proj)
        inception_4e_relu_5x5 = self.inception_4e_relu_5x5(inception_4e_5x5)
        inception_4e_relu_3x3 = self.inception_4e_relu_3x3(inception_4e_3x3)
        inception_4e_output = self.inception_4e_output((inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj), 1)
        pool4_3x3_s2_pad = F.pad(inception_4e_output, (0, 1, 0, 1), value=float('-inf'))
        pool4_3x3_s2 = self.pool4_3x3_s2(pool4_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_5a_1x1 = self.inception_5a_1x1(pool4_3x3_s2)
        inception_5a_5x5_reduce = self.inception_5a_5x5_reduce(pool4_3x3_s2)
        inception_5a_pool_pad = F.pad(pool4_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool = self.inception_5a_pool(inception_5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5a_3x3_reduce = self.inception_5a_3x3_reduce(pool4_3x3_s2)
        inception_5a_relu_1x1 = self.inception_5a_relu_1x1(inception_5a_1x1)
        inception_5a_relu_5x5_reduce = self.inception_5a_relu_5x5_reduce(inception_5a_5x5_reduce)
        inception_5a_pool_proj = self.inception_5a_pool_proj(inception_5a_pool)
        inception_5a_relu_3x3_reduce = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce)
        inception_5a_5x5_pad = F.pad(inception_5a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5a_5x5 = self.inception_5a_5x5(inception_5a_5x5_pad)
        inception_5a_relu_pool_proj = self.inception_5a_relu_pool_proj(inception_5a_pool_proj)
        inception_5a_3x3_pad = F.pad(inception_5a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5a_3x3 = self.inception_5a_3x3(inception_5a_3x3_pad)
        inception_5a_relu_5x5 = self.inception_5a_relu_5x5(inception_5a_5x5)
        inception_5a_relu_3x3 = self.inception_5a_relu_3x3(inception_5a_3x3)
        inception_5a_output = self.inception_5a_output((inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj), 1)
        inception_5b_pool_pad = F.pad(inception_5a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool = self.inception_5b_pool(inception_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5b_3x3_reduce = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_5x5_reduce = self.inception_5b_5x5_reduce(inception_5a_output)
        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)
        inception_5b_pool_proj = self.inception_5b_pool_proj(inception_5b_pool)
        inception_5b_relu_3x3_reduce = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce)
        inception_5b_relu_5x5_reduce = self.inception_5b_relu_5x5_reduce(inception_5b_5x5_reduce)
        inception_5b_relu_1x1 = self.inception_5b_relu_1x1(inception_5b_1x1)
        inception_5b_relu_pool_proj = self.inception_5b_relu_pool_proj(inception_5b_pool_proj)
        inception_5b_3x3_pad = F.pad(inception_5b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5b_3x3 = self.inception_5b_3x3(inception_5b_3x3_pad)
        inception_5b_5x5_pad = F.pad(inception_5b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5b_5x5 = self.inception_5b_5x5(inception_5b_5x5_pad)
        inception_5b_relu_3x3 = self.inception_5b_relu_3x3(inception_5b_3x3)
        inception_5b_relu_5x5 = self.inception_5b_relu_5x5(inception_5b_5x5)
        inception_5b_output = self.inception_5b_output((inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj), 1)
        pool5_7x7_s1 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        pool5_drop_7x7_s1 = self.pool5_drop_7x7_s1(input=pool5_7x7_s1, p=0.4000000059604645, training=self.training, inplace=True)
        return pool5_drop_7x7_s1


class GoogLeNetPlaces(nn.Module):

    def __init__(self, num_classes=205):
        super(GoogLeNetPlaces, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2_3x3_reduce = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_1x1 = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_1x1 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss1_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss1_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss1_classifier_1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_1x1 = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_1x1 = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loss2_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_5a_1x1 = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss2_classifier_1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_1x1 = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)

    def add_layers(self):
        self.conv1_relu_7x7 = helper_layers.ReluLayer()
        self.conv2_relu_3x3_reduce = helper_layers.ReluLayer()
        self.conv2_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_3a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_3b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_3b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_3b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4b_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4c_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4c_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4c_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4d_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4d_relu_3x3 = helper_layers.ReluLayer()
        self.inception_4d_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_1x1 = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_4e_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_4e_relu_5x5 = helper_layers.ReluLayer()
        self.inception_4e_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5a_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5a_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5a_relu_5x5 = helper_layers.ReluLayer()
        self.inception_5a_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5_reduce = helper_layers.ReluLayer()
        self.inception_5b_relu_1x1 = helper_layers.ReluLayer()
        self.inception_5b_relu_pool_proj = helper_layers.ReluLayer()
        self.inception_5b_relu_3x3 = helper_layers.ReluLayer()
        self.inception_5b_relu_5x5 = helper_layers.ReluLayer()
        self.pool1_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.pool2_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_3a_pool = helper_layers.MaxPool2dLayer()
        self.inception_3b_pool = helper_layers.MaxPool2dLayer()
        self.pool3_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_4a_pool = helper_layers.MaxPool2dLayer()
        self.inception_4b_pool = helper_layers.MaxPool2dLayer()
        self.inception_4c_pool = helper_layers.MaxPool2dLayer()
        self.inception_4d_pool = helper_layers.MaxPool2dLayer()
        self.inception_4e_pool = helper_layers.MaxPool2dLayer()
        self.pool4_3x3_s2 = helper_layers.MaxPool2dLayer()
        self.inception_5a_pool = helper_layers.MaxPool2dLayer()
        self.inception_5b_pool = helper_layers.MaxPool2dLayer()
        self.inception_3a_output = helper_layers.CatLayer()
        self.inception_3b_output = helper_layers.CatLayer()
        self.inception_4a_output = helper_layers.CatLayer()
        self.inception_4b_output = helper_layers.CatLayer()
        self.inception_4c_output = helper_layers.CatLayer()
        self.inception_4d_output = helper_layers.CatLayer()
        self.inception_4e_output = helper_layers.CatLayer()
        self.inception_5a_output = helper_layers.CatLayer()
        self.inception_5b_output = helper_layers.CatLayer()
        self.pool5_drop_7x7_s1 = helper_layers.DropoutLayer()

    def forward(self, x):
        conv1_7x7_s2_pad = F.pad(x, (3, 3, 3, 3))
        conv1_7x7_s2 = self.conv1_7x7_s2(conv1_7x7_s2_pad)
        conv1_relu_7x7 = self.conv1_relu_7x7(conv1_7x7_s2)
        pool1_3x3_s2_pad = F.pad(conv1_relu_7x7, (0, 1, 0, 1), value=float('-inf'))
        pool1_3x3_s2 = self.pool1_3x3_s2(pool1_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        pool1_norm1 = F.local_response_norm(pool1_3x3_s2, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_3x3_reduce = self.conv2_3x3_reduce(pool1_norm1)
        conv2_relu_3x3_reduce = self.conv2_relu_3x3_reduce(conv2_3x3_reduce)
        conv2_3x3_pad = F.pad(conv2_relu_3x3_reduce, (1, 1, 1, 1))
        conv2_3x3 = self.conv2_3x3(conv2_3x3_pad)
        conv2_relu_3x3 = self.conv2_relu_3x3(conv2_3x3)
        conv2_norm2 = F.local_response_norm(conv2_relu_3x3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_3x3_s2_pad = F.pad(conv2_norm2, (0, 1, 0, 1), value=float('-inf'))
        pool2_3x3_s2 = self.pool2_3x3_s2(pool2_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_pool_pad = F.pad(pool2_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool = self.inception_3a_pool(inception_3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3a_1x1 = self.inception_3a_1x1(pool2_3x3_s2)
        inception_3a_5x5_reduce = self.inception_3a_5x5_reduce(pool2_3x3_s2)
        inception_3a_3x3_reduce = self.inception_3a_3x3_reduce(pool2_3x3_s2)
        inception_3a_pool_proj = self.inception_3a_pool_proj(inception_3a_pool)
        inception_3a_relu_1x1 = self.inception_3a_relu_1x1(inception_3a_1x1)
        inception_3a_relu_5x5_reduce = self.inception_3a_relu_5x5_reduce(inception_3a_5x5_reduce)
        inception_3a_relu_3x3_reduce = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce)
        inception_3a_relu_pool_proj = self.inception_3a_relu_pool_proj(inception_3a_pool_proj)
        inception_3a_5x5_pad = F.pad(inception_3a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3a_5x5 = self.inception_3a_5x5(inception_3a_5x5_pad)
        inception_3a_3x3_pad = F.pad(inception_3a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3a_3x3 = self.inception_3a_3x3(inception_3a_3x3_pad)
        inception_3a_relu_5x5 = self.inception_3a_relu_5x5(inception_3a_5x5)
        inception_3a_relu_3x3 = self.inception_3a_relu_3x3(inception_3a_3x3)
        inception_3a_output = self.inception_3a_output((inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj), 1)
        inception_3b_3x3_reduce = self.inception_3b_3x3_reduce(inception_3a_output)
        inception_3b_pool_pad = F.pad(inception_3a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool = self.inception_3b_pool(inception_3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_3b_1x1 = self.inception_3b_1x1(inception_3a_output)
        inception_3b_5x5_reduce = self.inception_3b_5x5_reduce(inception_3a_output)
        inception_3b_relu_3x3_reduce = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce)
        inception_3b_pool_proj = self.inception_3b_pool_proj(inception_3b_pool)
        inception_3b_relu_1x1 = self.inception_3b_relu_1x1(inception_3b_1x1)
        inception_3b_relu_5x5_reduce = self.inception_3b_relu_5x5_reduce(inception_3b_5x5_reduce)
        inception_3b_3x3_pad = F.pad(inception_3b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3b_3x3 = self.inception_3b_3x3(inception_3b_3x3_pad)
        inception_3b_relu_pool_proj = self.inception_3b_relu_pool_proj(inception_3b_pool_proj)
        inception_3b_5x5_pad = F.pad(inception_3b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3b_5x5 = self.inception_3b_5x5(inception_3b_5x5_pad)
        inception_3b_relu_3x3 = self.inception_3b_relu_3x3(inception_3b_3x3)
        inception_3b_relu_5x5 = self.inception_3b_relu_5x5(inception_3b_5x5)
        inception_3b_output = self.inception_3b_output((inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj), 1)
        pool3_3x3_s2_pad = F.pad(inception_3b_output, (0, 1, 0, 1), value=float('-inf'))
        pool3_3x3_s2 = self.pool3_3x3_s2(pool3_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_1x1 = self.inception_4a_1x1(pool3_3x3_s2)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(pool3_3x3_s2)
        inception_4a_5x5_reduce = self.inception_4a_5x5_reduce(pool3_3x3_s2)
        inception_4a_pool_pad = F.pad(pool3_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool = self.inception_4a_pool(inception_4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4a_relu_1x1 = self.inception_4a_relu_1x1(inception_4a_1x1)
        inception_4a_relu_3x3_reduce = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce)
        inception_4a_relu_5x5_reduce = self.inception_4a_relu_5x5_reduce(inception_4a_5x5_reduce)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        inception_4a_3x3_pad = F.pad(inception_4a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_pad)
        inception_4a_5x5_pad = F.pad(inception_4a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4a_5x5 = self.inception_4a_5x5(inception_4a_5x5_pad)
        inception_4a_relu_pool_proj = self.inception_4a_relu_pool_proj(inception_4a_pool_proj)
        inception_4a_relu_3x3 = self.inception_4a_relu_3x3(inception_4a_3x3)
        inception_4a_relu_5x5 = self.inception_4a_relu_5x5(inception_4a_5x5)
        inception_4a_output = self.inception_4a_output((inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj), 1)
        inception_4b_pool_pad = F.pad(inception_4a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool = self.inception_4b_pool(inception_4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4b_5x5_reduce = self.inception_4b_5x5_reduce(inception_4a_output)
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        inception_4b_relu_5x5_reduce = self.inception_4b_relu_5x5_reduce(inception_4b_5x5_reduce)
        inception_4b_relu_1x1 = self.inception_4b_relu_1x1(inception_4b_1x1)
        inception_4b_relu_3x3_reduce = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce)
        inception_4b_relu_pool_proj = self.inception_4b_relu_pool_proj(inception_4b_pool_proj)
        inception_4b_5x5_pad = F.pad(inception_4b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4b_5x5 = self.inception_4b_5x5(inception_4b_5x5_pad)
        inception_4b_3x3_pad = F.pad(inception_4b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_pad)
        inception_4b_relu_5x5 = self.inception_4b_relu_5x5(inception_4b_5x5)
        inception_4b_relu_3x3 = self.inception_4b_relu_3x3(inception_4b_3x3)
        inception_4b_output = self.inception_4b_output((inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj), 1)
        inception_4c_5x5_reduce = self.inception_4c_5x5_reduce(inception_4b_output)
        inception_4c_pool_pad = F.pad(inception_4b_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool = self.inception_4c_pool(inception_4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_relu_5x5_reduce = self.inception_4c_relu_5x5_reduce(inception_4c_5x5_reduce)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_relu_1x1 = self.inception_4c_relu_1x1(inception_4c_1x1)
        inception_4c_relu_3x3_reduce = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce)
        inception_4c_5x5_pad = F.pad(inception_4c_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4c_5x5 = self.inception_4c_5x5(inception_4c_5x5_pad)
        inception_4c_relu_pool_proj = self.inception_4c_relu_pool_proj(inception_4c_pool_proj)
        inception_4c_3x3_pad = F.pad(inception_4c_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_pad)
        inception_4c_relu_5x5 = self.inception_4c_relu_5x5(inception_4c_5x5)
        inception_4c_relu_3x3 = self.inception_4c_relu_3x3(inception_4c_3x3)
        inception_4c_output = self.inception_4c_output((inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj), 1)
        inception_4d_pool_pad = F.pad(inception_4c_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool = self.inception_4d_pool(inception_4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_5x5_reduce = self.inception_4d_5x5_reduce(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_relu_3x3_reduce = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce)
        inception_4d_relu_1x1 = self.inception_4d_relu_1x1(inception_4d_1x1)
        inception_4d_relu_5x5_reduce = self.inception_4d_relu_5x5_reduce(inception_4d_5x5_reduce)
        inception_4d_relu_pool_proj = self.inception_4d_relu_pool_proj(inception_4d_pool_proj)
        inception_4d_3x3_pad = F.pad(inception_4d_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_pad)
        inception_4d_5x5_pad = F.pad(inception_4d_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4d_5x5 = self.inception_4d_5x5(inception_4d_5x5_pad)
        inception_4d_relu_3x3 = self.inception_4d_relu_3x3(inception_4d_3x3)
        inception_4d_relu_5x5 = self.inception_4d_relu_5x5(inception_4d_5x5)
        inception_4d_output = self.inception_4d_output((inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj), 1)
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_5x5_reduce = self.inception_4e_5x5_reduce(inception_4d_output)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_pool_pad = F.pad(inception_4d_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool = self.inception_4e_pool(inception_4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_4e_relu_1x1 = self.inception_4e_relu_1x1(inception_4e_1x1)
        inception_4e_relu_5x5_reduce = self.inception_4e_relu_5x5_reduce(inception_4e_5x5_reduce)
        inception_4e_relu_3x3_reduce = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce)
        inception_4e_pool_proj = self.inception_4e_pool_proj(inception_4e_pool)
        inception_4e_5x5_pad = F.pad(inception_4e_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4e_5x5 = self.inception_4e_5x5(inception_4e_5x5_pad)
        inception_4e_3x3_pad = F.pad(inception_4e_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4e_3x3 = self.inception_4e_3x3(inception_4e_3x3_pad)
        inception_4e_relu_pool_proj = self.inception_4e_relu_pool_proj(inception_4e_pool_proj)
        inception_4e_relu_5x5 = self.inception_4e_relu_5x5(inception_4e_5x5)
        inception_4e_relu_3x3 = self.inception_4e_relu_3x3(inception_4e_3x3)
        inception_4e_output = self.inception_4e_output((inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj), 1)
        pool4_3x3_s2_pad = F.pad(inception_4e_output, (0, 1, 0, 1), value=float('-inf'))
        pool4_3x3_s2 = self.pool4_3x3_s2(pool4_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_5a_1x1 = self.inception_5a_1x1(pool4_3x3_s2)
        inception_5a_5x5_reduce = self.inception_5a_5x5_reduce(pool4_3x3_s2)
        inception_5a_pool_pad = F.pad(pool4_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool = self.inception_5a_pool(inception_5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5a_3x3_reduce = self.inception_5a_3x3_reduce(pool4_3x3_s2)
        inception_5a_relu_1x1 = self.inception_5a_relu_1x1(inception_5a_1x1)
        inception_5a_relu_5x5_reduce = self.inception_5a_relu_5x5_reduce(inception_5a_5x5_reduce)
        inception_5a_pool_proj = self.inception_5a_pool_proj(inception_5a_pool)
        inception_5a_relu_3x3_reduce = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce)
        inception_5a_5x5_pad = F.pad(inception_5a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5a_5x5 = self.inception_5a_5x5(inception_5a_5x5_pad)
        inception_5a_relu_pool_proj = self.inception_5a_relu_pool_proj(inception_5a_pool_proj)
        inception_5a_3x3_pad = F.pad(inception_5a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5a_3x3 = self.inception_5a_3x3(inception_5a_3x3_pad)
        inception_5a_relu_5x5 = self.inception_5a_relu_5x5(inception_5a_5x5)
        inception_5a_relu_3x3 = self.inception_5a_relu_3x3(inception_5a_3x3)
        inception_5a_output = self.inception_5a_output((inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj), 1)
        inception_5b_3x3_reduce = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_pool_pad = F.pad(inception_5a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool = self.inception_5b_pool(inception_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        inception_5b_5x5_reduce = self.inception_5b_5x5_reduce(inception_5a_output)
        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)
        inception_5b_relu_3x3_reduce = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce)
        inception_5b_pool_proj = self.inception_5b_pool_proj(inception_5b_pool)
        inception_5b_relu_5x5_reduce = self.inception_5b_relu_5x5_reduce(inception_5b_5x5_reduce)
        inception_5b_relu_1x1 = self.inception_5b_relu_1x1(inception_5b_1x1)
        inception_5b_3x3_pad = F.pad(inception_5b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5b_3x3 = self.inception_5b_3x3(inception_5b_3x3_pad)
        inception_5b_relu_pool_proj = self.inception_5b_relu_pool_proj(inception_5b_pool_proj)
        inception_5b_5x5_pad = F.pad(inception_5b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5b_5x5 = self.inception_5b_5x5(inception_5b_5x5_pad)
        inception_5b_relu_3x3 = self.inception_5b_relu_3x3(inception_5b_3x3)
        inception_5b_relu_5x5 = self.inception_5b_relu_5x5(inception_5b_5x5)
        inception_5b_output = self.inception_5b_output((inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj), 1)
        pool5_7x7_s1 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        pool5_drop_7x7_s1 = self.pool5_drop_7x7_s1(input=pool5_7x7_s1, p=0.4000000059604645, training=False, inplace=True)
        return pool5_drop_7x7_s1


class Inception5h(nn.Module):

    def __init__(self):
        super(Inception5h, self).__init__()
        self.conv2d0_pre_relu_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2d1_pre_relu_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2d2_pre_relu_conv = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_1x1_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_3x3_pre_relu_conv = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_5x5_pre_relu_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_1x1_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_3x3_pre_relu_conv = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_1x1_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_3x3_pre_relu_conv = nn.Conv2d(in_channels=96, out_channels=204, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_5x5_pre_relu_conv = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_1x1_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_3x3_pre_relu_conv = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_5x5_pre_relu_conv = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_1x1_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_3x3_pre_relu_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_5x5_pre_relu_conv = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_1x1_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_3x3_pre_relu_conv = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_1x1_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_3x3_pre_relu_conv = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_1x1_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_3x3_pre_relu_conv = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_5x5_pre_relu_conv = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_1x1_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_3x3_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_5x5_pre_relu_conv = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.softmax2_pre_activation_matmul = nn.Linear(in_features=1024, out_features=1008, bias=True)

    def add_layers(self):
        self.conv2d0 = helper_layers.ReluLayer()
        self.maxpool0 = helper_layers.MaxPool2dLayer()
        self.conv2d1 = helper_layers.ReluLayer()
        self.conv2d2 = helper_layers.ReluLayer()
        self.maxpool1 = helper_layers.MaxPool2dLayer()
        self.mixed3a_pool = helper_layers.MaxPool2dLayer()
        self.mixed3a_1x1 = helper_layers.ReluLayer()
        self.mixed3a_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed3a_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed3a_pool_reduce = helper_layers.ReluLayer()
        self.mixed3a_3x3 = helper_layers.ReluLayer()
        self.mixed3a_5x5 = helper_layers.ReluLayer()
        self.mixed3a = helper_layers.CatLayer()
        self.mixed3b_pool = helper_layers.MaxPool2dLayer()
        self.mixed3b_1x1 = helper_layers.ReluLayer()
        self.mixed3b_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed3b_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed3b_pool_reduce = helper_layers.ReluLayer()
        self.mixed3b_3x3 = helper_layers.ReluLayer()
        self.mixed3b_5x5 = helper_layers.ReluLayer()
        self.mixed3b = helper_layers.CatLayer()
        self.maxpool4 = helper_layers.MaxPool2dLayer()
        self.mixed4a_pool = helper_layers.MaxPool2dLayer()
        self.mixed4a_1x1 = helper_layers.ReluLayer()
        self.mixed4a_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed4a_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed4a_pool_reduce = helper_layers.ReluLayer()
        self.mixed4a_3x3 = helper_layers.ReluLayer()
        self.mixed4a_5x5 = helper_layers.ReluLayer()
        self.mixed4a = helper_layers.CatLayer()
        self.mixed4b_pool = helper_layers.MaxPool2dLayer()
        self.mixed4b_1x1 = helper_layers.ReluLayer()
        self.mixed4b_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed4b_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed4b_pool_reduce = helper_layers.ReluLayer()
        self.mixed4b_3x3 = helper_layers.ReluLayer()
        self.mixed4b_5x5 = helper_layers.ReluLayer()
        self.mixed4b = helper_layers.CatLayer()
        self.mixed4c_pool = helper_layers.MaxPool2dLayer()
        self.mixed4c_1x1 = helper_layers.ReluLayer()
        self.mixed4c_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed4c_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed4c_pool_reduce = helper_layers.ReluLayer()
        self.mixed4c_3x3 = helper_layers.ReluLayer()
        self.mixed4c_5x5 = helper_layers.ReluLayer()
        self.mixed4c = helper_layers.CatLayer()
        self.mixed4d_pool = helper_layers.MaxPool2dLayer()
        self.mixed4d_1x1 = helper_layers.ReluLayer()
        self.mixed4d_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed4d_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed4d_pool_reduce = helper_layers.ReluLayer()
        self.mixed4d_3x3 = helper_layers.ReluLayer()
        self.mixed4d_5x5 = helper_layers.ReluLayer()
        self.mixed4d = helper_layers.CatLayer()
        self.mixed4e_pool = helper_layers.MaxPool2dLayer()
        self.mixed4e_1x1 = helper_layers.ReluLayer()
        self.mixed4e_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed4e_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed4e_pool_reduce = helper_layers.ReluLayer()
        self.mixed4e_3x3 = helper_layers.ReluLayer()
        self.mixed4e_5x5 = helper_layers.ReluLayer()
        self.mixed4e = helper_layers.CatLayer()
        self.maxpool10 = helper_layers.MaxPool2dLayer()
        self.mixed5a_pool = helper_layers.MaxPool2dLayer()
        self.mixed5a_1x1 = helper_layers.ReluLayer()
        self.mixed5a_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed5a_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed5a_pool_reduce = helper_layers.ReluLayer()
        self.mixed5a_3x3 = helper_layers.ReluLayer()
        self.mixed5a_5x5 = helper_layers.ReluLayer()
        self.mixed5a = helper_layers.CatLayer()
        self.mixed5b_pool = helper_layers.MaxPool2dLayer()
        self.mixed5b_1x1 = helper_layers.ReluLayer()
        self.mixed5b_3x3_bottleneck = helper_layers.ReluLayer()
        self.mixed5b_5x5_bottleneck = helper_layers.ReluLayer()
        self.mixed5b_pool_reduce = helper_layers.ReluLayer()
        self.mixed5b_3x3 = helper_layers.ReluLayer()
        self.mixed5b_5x5 = helper_layers.ReluLayer()
        self.mixed5b = helper_layers.CatLayer()
        self.softmax2 = helper_layers.SoftMaxLayer()

    def forward(self, x):
        conv2d0_pre_relu_conv_pad = F.pad(x, (2, 3, 2, 3))
        conv2d0_pre_relu_conv = self.conv2d0_pre_relu_conv(conv2d0_pre_relu_conv_pad)
        conv2d0 = self.conv2d0(conv2d0_pre_relu_conv)
        maxpool0_pad = F.pad(conv2d0, (0, 1, 0, 1), value=float('-inf'))
        maxpool0 = self.maxpool0(maxpool0_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        localresponsenorm0 = F.local_response_norm(maxpool0, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
        conv2d1_pre_relu_conv = self.conv2d1_pre_relu_conv(localresponsenorm0)
        conv2d1 = self.conv2d1(conv2d1_pre_relu_conv)
        conv2d2_pre_relu_conv_pad = F.pad(conv2d1, (1, 1, 1, 1))
        conv2d2_pre_relu_conv = self.conv2d2_pre_relu_conv(conv2d2_pre_relu_conv_pad)
        conv2d2 = self.conv2d2(conv2d2_pre_relu_conv)
        localresponsenorm1 = F.local_response_norm(conv2d2, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
        maxpool1_pad = F.pad(localresponsenorm1, (0, 1, 0, 1), value=float('-inf'))
        maxpool1 = self.maxpool1(maxpool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed3a_1x1_pre_relu_conv = self.mixed3a_1x1_pre_relu_conv(maxpool1)
        mixed3a_3x3_bottleneck_pre_relu_conv = self.mixed3a_3x3_bottleneck_pre_relu_conv(maxpool1)
        mixed3a_5x5_bottleneck_pre_relu_conv = self.mixed3a_5x5_bottleneck_pre_relu_conv(maxpool1)
        mixed3a_pool_pad = F.pad(maxpool1, (1, 1, 1, 1), value=float('-inf'))
        mixed3a_pool = self.mixed3a_pool(mixed3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed3a_1x1 = self.mixed3a_1x1(mixed3a_1x1_pre_relu_conv)
        mixed3a_3x3_bottleneck = self.mixed3a_3x3_bottleneck(mixed3a_3x3_bottleneck_pre_relu_conv)
        mixed3a_5x5_bottleneck = self.mixed3a_5x5_bottleneck(mixed3a_5x5_bottleneck_pre_relu_conv)
        mixed3a_pool_reduce_pre_relu_conv = self.mixed3a_pool_reduce_pre_relu_conv(mixed3a_pool)
        mixed3a_3x3_pre_relu_conv_pad = F.pad(mixed3a_3x3_bottleneck, (1, 1, 1, 1))
        mixed3a_3x3_pre_relu_conv = self.mixed3a_3x3_pre_relu_conv(mixed3a_3x3_pre_relu_conv_pad)
        mixed3a_5x5_pre_relu_conv_pad = F.pad(mixed3a_5x5_bottleneck, (2, 2, 2, 2))
        mixed3a_5x5_pre_relu_conv = self.mixed3a_5x5_pre_relu_conv(mixed3a_5x5_pre_relu_conv_pad)
        mixed3a_pool_reduce = self.mixed3a_pool_reduce(mixed3a_pool_reduce_pre_relu_conv)
        mixed3a_3x3 = self.mixed3a_3x3(mixed3a_3x3_pre_relu_conv)
        mixed3a_5x5 = self.mixed3a_5x5(mixed3a_5x5_pre_relu_conv)
        mixed3a = self.mixed3a((mixed3a_1x1, mixed3a_3x3, mixed3a_5x5, mixed3a_pool_reduce), 1)
        mixed3b_1x1_pre_relu_conv = self.mixed3b_1x1_pre_relu_conv(mixed3a)
        mixed3b_3x3_bottleneck_pre_relu_conv = self.mixed3b_3x3_bottleneck_pre_relu_conv(mixed3a)
        mixed3b_5x5_bottleneck_pre_relu_conv = self.mixed3b_5x5_bottleneck_pre_relu_conv(mixed3a)
        mixed3b_pool_pad = F.pad(mixed3a, (1, 1, 1, 1), value=float('-inf'))
        mixed3b_pool = self.mixed3b_pool(mixed3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed3b_1x1 = self.mixed3b_1x1(mixed3b_1x1_pre_relu_conv)
        mixed3b_3x3_bottleneck = self.mixed3b_3x3_bottleneck(mixed3b_3x3_bottleneck_pre_relu_conv)
        mixed3b_5x5_bottleneck = self.mixed3b_5x5_bottleneck(mixed3b_5x5_bottleneck_pre_relu_conv)
        mixed3b_pool_reduce_pre_relu_conv = self.mixed3b_pool_reduce_pre_relu_conv(mixed3b_pool)
        mixed3b_3x3_pre_relu_conv_pad = F.pad(mixed3b_3x3_bottleneck, (1, 1, 1, 1))
        mixed3b_3x3_pre_relu_conv = self.mixed3b_3x3_pre_relu_conv(mixed3b_3x3_pre_relu_conv_pad)
        mixed3b_5x5_pre_relu_conv_pad = F.pad(mixed3b_5x5_bottleneck, (2, 2, 2, 2))
        mixed3b_5x5_pre_relu_conv = self.mixed3b_5x5_pre_relu_conv(mixed3b_5x5_pre_relu_conv_pad)
        mixed3b_pool_reduce = self.mixed3b_pool_reduce(mixed3b_pool_reduce_pre_relu_conv)
        mixed3b_3x3 = self.mixed3b_3x3(mixed3b_3x3_pre_relu_conv)
        mixed3b_5x5 = self.mixed3b_5x5(mixed3b_5x5_pre_relu_conv)
        mixed3b = self.mixed3b((mixed3b_1x1, mixed3b_3x3, mixed3b_5x5, mixed3b_pool_reduce), 1)
        maxpool4_pad = F.pad(mixed3b, (0, 1, 0, 1), value=float('-inf'))
        maxpool4 = self.maxpool4(maxpool4_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed4a_1x1_pre_relu_conv = self.mixed4a_1x1_pre_relu_conv(maxpool4)
        mixed4a_3x3_bottleneck_pre_relu_conv = self.mixed4a_3x3_bottleneck_pre_relu_conv(maxpool4)
        mixed4a_5x5_bottleneck_pre_relu_conv = self.mixed4a_5x5_bottleneck_pre_relu_conv(maxpool4)
        mixed4a_pool_pad = F.pad(maxpool4, (1, 1, 1, 1), value=float('-inf'))
        mixed4a_pool = self.mixed4a_pool(mixed4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4a_1x1 = self.mixed4a_1x1(mixed4a_1x1_pre_relu_conv)
        mixed4a_3x3_bottleneck = self.mixed4a_3x3_bottleneck(mixed4a_3x3_bottleneck_pre_relu_conv)
        mixed4a_5x5_bottleneck = self.mixed4a_5x5_bottleneck(mixed4a_5x5_bottleneck_pre_relu_conv)
        mixed4a_pool_reduce_pre_relu_conv = self.mixed4a_pool_reduce_pre_relu_conv(mixed4a_pool)
        mixed4a_3x3_pre_relu_conv_pad = F.pad(mixed4a_3x3_bottleneck, (1, 1, 1, 1))
        mixed4a_3x3_pre_relu_conv = self.mixed4a_3x3_pre_relu_conv(mixed4a_3x3_pre_relu_conv_pad)
        mixed4a_5x5_pre_relu_conv_pad = F.pad(mixed4a_5x5_bottleneck, (2, 2, 2, 2))
        mixed4a_5x5_pre_relu_conv = self.mixed4a_5x5_pre_relu_conv(mixed4a_5x5_pre_relu_conv_pad)
        mixed4a_pool_reduce = self.mixed4a_pool_reduce(mixed4a_pool_reduce_pre_relu_conv)
        mixed4a_3x3 = self.mixed4a_3x3(mixed4a_3x3_pre_relu_conv)
        mixed4a_5x5 = self.mixed4a_5x5(mixed4a_5x5_pre_relu_conv)
        mixed4a = self.mixed4a((mixed4a_1x1, mixed4a_3x3, mixed4a_5x5, mixed4a_pool_reduce), 1)
        mixed4b_1x1_pre_relu_conv = self.mixed4b_1x1_pre_relu_conv(mixed4a)
        mixed4b_3x3_bottleneck_pre_relu_conv = self.mixed4b_3x3_bottleneck_pre_relu_conv(mixed4a)
        mixed4b_5x5_bottleneck_pre_relu_conv = self.mixed4b_5x5_bottleneck_pre_relu_conv(mixed4a)
        mixed4b_pool_pad = F.pad(mixed4a, (1, 1, 1, 1), value=float('-inf'))
        mixed4b_pool = self.mixed4b_pool(mixed4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4b_1x1 = self.mixed4b_1x1(mixed4b_1x1_pre_relu_conv)
        mixed4b_3x3_bottleneck = self.mixed4b_3x3_bottleneck(mixed4b_3x3_bottleneck_pre_relu_conv)
        mixed4b_5x5_bottleneck = self.mixed4b_5x5_bottleneck(mixed4b_5x5_bottleneck_pre_relu_conv)
        mixed4b_pool_reduce_pre_relu_conv = self.mixed4b_pool_reduce_pre_relu_conv(mixed4b_pool)
        mixed4b_3x3_pre_relu_conv_pad = F.pad(mixed4b_3x3_bottleneck, (1, 1, 1, 1))
        mixed4b_3x3_pre_relu_conv = self.mixed4b_3x3_pre_relu_conv(mixed4b_3x3_pre_relu_conv_pad)
        mixed4b_5x5_pre_relu_conv_pad = F.pad(mixed4b_5x5_bottleneck, (2, 2, 2, 2))
        mixed4b_5x5_pre_relu_conv = self.mixed4b_5x5_pre_relu_conv(mixed4b_5x5_pre_relu_conv_pad)
        mixed4b_pool_reduce = self.mixed4b_pool_reduce(mixed4b_pool_reduce_pre_relu_conv)
        mixed4b_3x3 = self.mixed4b_3x3(mixed4b_3x3_pre_relu_conv)
        mixed4b_5x5 = self.mixed4b_5x5(mixed4b_5x5_pre_relu_conv)
        mixed4b = self.mixed4b((mixed4b_1x1, mixed4b_3x3, mixed4b_5x5, mixed4b_pool_reduce), 1)
        mixed4c_1x1_pre_relu_conv = self.mixed4c_1x1_pre_relu_conv(mixed4b)
        mixed4c_3x3_bottleneck_pre_relu_conv = self.mixed4c_3x3_bottleneck_pre_relu_conv(mixed4b)
        mixed4c_5x5_bottleneck_pre_relu_conv = self.mixed4c_5x5_bottleneck_pre_relu_conv(mixed4b)
        mixed4c_pool_pad = F.pad(mixed4b, (1, 1, 1, 1), value=float('-inf'))
        mixed4c_pool = self.mixed4c_pool(mixed4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4c_1x1 = self.mixed4c_1x1(mixed4c_1x1_pre_relu_conv)
        mixed4c_3x3_bottleneck = self.mixed4c_3x3_bottleneck(mixed4c_3x3_bottleneck_pre_relu_conv)
        mixed4c_5x5_bottleneck = self.mixed4c_5x5_bottleneck(mixed4c_5x5_bottleneck_pre_relu_conv)
        mixed4c_pool_reduce_pre_relu_conv = self.mixed4c_pool_reduce_pre_relu_conv(mixed4c_pool)
        mixed4c_3x3_pre_relu_conv_pad = F.pad(mixed4c_3x3_bottleneck, (1, 1, 1, 1))
        mixed4c_3x3_pre_relu_conv = self.mixed4c_3x3_pre_relu_conv(mixed4c_3x3_pre_relu_conv_pad)
        mixed4c_5x5_pre_relu_conv_pad = F.pad(mixed4c_5x5_bottleneck, (2, 2, 2, 2))
        mixed4c_5x5_pre_relu_conv = self.mixed4c_5x5_pre_relu_conv(mixed4c_5x5_pre_relu_conv_pad)
        mixed4c_pool_reduce = self.mixed4c_pool_reduce(mixed4c_pool_reduce_pre_relu_conv)
        mixed4c_3x3 = self.mixed4c_3x3(mixed4c_3x3_pre_relu_conv)
        mixed4c_5x5 = self.mixed4c_5x5(mixed4c_5x5_pre_relu_conv)
        mixed4c = self.mixed4c((mixed4c_1x1, mixed4c_3x3, mixed4c_5x5, mixed4c_pool_reduce), 1)
        mixed4d_1x1_pre_relu_conv = self.mixed4d_1x1_pre_relu_conv(mixed4c)
        mixed4d_3x3_bottleneck_pre_relu_conv = self.mixed4d_3x3_bottleneck_pre_relu_conv(mixed4c)
        mixed4d_5x5_bottleneck_pre_relu_conv = self.mixed4d_5x5_bottleneck_pre_relu_conv(mixed4c)
        mixed4d_pool_pad = F.pad(mixed4c, (1, 1, 1, 1), value=float('-inf'))
        mixed4d_pool = self.mixed4d_pool(mixed4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4d_1x1 = self.mixed4d_1x1(mixed4d_1x1_pre_relu_conv)
        mixed4d_3x3_bottleneck = self.mixed4d_3x3_bottleneck(mixed4d_3x3_bottleneck_pre_relu_conv)
        mixed4d_5x5_bottleneck = self.mixed4d_5x5_bottleneck(mixed4d_5x5_bottleneck_pre_relu_conv)
        mixed4d_pool_reduce_pre_relu_conv = self.mixed4d_pool_reduce_pre_relu_conv(mixed4d_pool)
        mixed4d_3x3_pre_relu_conv_pad = F.pad(mixed4d_3x3_bottleneck, (1, 1, 1, 1))
        mixed4d_3x3_pre_relu_conv = self.mixed4d_3x3_pre_relu_conv(mixed4d_3x3_pre_relu_conv_pad)
        mixed4d_5x5_pre_relu_conv_pad = F.pad(mixed4d_5x5_bottleneck, (2, 2, 2, 2))
        mixed4d_5x5_pre_relu_conv = self.mixed4d_5x5_pre_relu_conv(mixed4d_5x5_pre_relu_conv_pad)
        mixed4d_pool_reduce = self.mixed4d_pool_reduce(mixed4d_pool_reduce_pre_relu_conv)
        mixed4d_3x3 = self.mixed4d_3x3(mixed4d_3x3_pre_relu_conv)
        mixed4d_5x5 = self.mixed4d_5x5(mixed4d_5x5_pre_relu_conv)
        mixed4d = self.mixed4d((mixed4d_1x1, mixed4d_3x3, mixed4d_5x5, mixed4d_pool_reduce), 1)
        mixed4e_1x1_pre_relu_conv = self.mixed4e_1x1_pre_relu_conv(mixed4d)
        mixed4e_3x3_bottleneck_pre_relu_conv = self.mixed4e_3x3_bottleneck_pre_relu_conv(mixed4d)
        mixed4e_5x5_bottleneck_pre_relu_conv = self.mixed4e_5x5_bottleneck_pre_relu_conv(mixed4d)
        mixed4e_pool_pad = F.pad(mixed4d, (1, 1, 1, 1), value=float('-inf'))
        mixed4e_pool = self.mixed4e_pool(mixed4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4e_1x1 = self.mixed4e_1x1(mixed4e_1x1_pre_relu_conv)
        mixed4e_3x3_bottleneck = self.mixed4e_3x3_bottleneck(mixed4e_3x3_bottleneck_pre_relu_conv)
        mixed4e_5x5_bottleneck = self.mixed4e_5x5_bottleneck(mixed4e_5x5_bottleneck_pre_relu_conv)
        mixed4e_pool_reduce_pre_relu_conv = self.mixed4e_pool_reduce_pre_relu_conv(mixed4e_pool)
        mixed4e_3x3_pre_relu_conv_pad = F.pad(mixed4e_3x3_bottleneck, (1, 1, 1, 1))
        mixed4e_3x3_pre_relu_conv = self.mixed4e_3x3_pre_relu_conv(mixed4e_3x3_pre_relu_conv_pad)
        mixed4e_5x5_pre_relu_conv_pad = F.pad(mixed4e_5x5_bottleneck, (2, 2, 2, 2))
        mixed4e_5x5_pre_relu_conv = self.mixed4e_5x5_pre_relu_conv(mixed4e_5x5_pre_relu_conv_pad)
        mixed4e_pool_reduce = self.mixed4e_pool_reduce(mixed4e_pool_reduce_pre_relu_conv)
        mixed4e_3x3 = self.mixed4e_3x3(mixed4e_3x3_pre_relu_conv)
        mixed4e_5x5 = self.mixed4e_5x5(mixed4e_5x5_pre_relu_conv)
        mixed4e = self.mixed4e((mixed4e_1x1, mixed4e_3x3, mixed4e_5x5, mixed4e_pool_reduce), 1)
        maxpool10_pad = F.pad(mixed4e, (0, 1, 0, 1), value=float('-inf'))
        maxpool10 = self.maxpool10(maxpool10_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed5a_1x1_pre_relu_conv = self.mixed5a_1x1_pre_relu_conv(maxpool10)
        mixed5a_3x3_bottleneck_pre_relu_conv = self.mixed5a_3x3_bottleneck_pre_relu_conv(maxpool10)
        mixed5a_5x5_bottleneck_pre_relu_conv = self.mixed5a_5x5_bottleneck_pre_relu_conv(maxpool10)
        mixed5a_pool_pad = F.pad(maxpool10, (1, 1, 1, 1), value=float('-inf'))
        mixed5a_pool = self.mixed5a_pool(mixed5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed5a_1x1 = self.mixed5a_1x1(mixed5a_1x1_pre_relu_conv)
        mixed5a_3x3_bottleneck = self.mixed5a_3x3_bottleneck(mixed5a_3x3_bottleneck_pre_relu_conv)
        mixed5a_5x5_bottleneck = self.mixed5a_5x5_bottleneck(mixed5a_5x5_bottleneck_pre_relu_conv)
        mixed5a_pool_reduce_pre_relu_conv = self.mixed5a_pool_reduce_pre_relu_conv(mixed5a_pool)
        mixed5a_3x3_pre_relu_conv_pad = F.pad(mixed5a_3x3_bottleneck, (1, 1, 1, 1))
        mixed5a_3x3_pre_relu_conv = self.mixed5a_3x3_pre_relu_conv(mixed5a_3x3_pre_relu_conv_pad)
        mixed5a_5x5_pre_relu_conv_pad = F.pad(mixed5a_5x5_bottleneck, (2, 2, 2, 2))
        mixed5a_5x5_pre_relu_conv = self.mixed5a_5x5_pre_relu_conv(mixed5a_5x5_pre_relu_conv_pad)
        mixed5a_pool_reduce = self.mixed5a_pool_reduce(mixed5a_pool_reduce_pre_relu_conv)
        mixed5a_3x3 = self.mixed5a_3x3(mixed5a_3x3_pre_relu_conv)
        mixed5a_5x5 = self.mixed5a_5x5(mixed5a_5x5_pre_relu_conv)
        mixed5a = self.mixed5a((mixed5a_1x1, mixed5a_3x3, mixed5a_5x5, mixed5a_pool_reduce), 1)
        mixed5b_1x1_pre_relu_conv = self.mixed5b_1x1_pre_relu_conv(mixed5a)
        mixed5b_3x3_bottleneck_pre_relu_conv = self.mixed5b_3x3_bottleneck_pre_relu_conv(mixed5a)
        mixed5b_5x5_bottleneck_pre_relu_conv = self.mixed5b_5x5_bottleneck_pre_relu_conv(mixed5a)
        mixed5b_pool_pad = F.pad(mixed5a, (1, 1, 1, 1), value=float('-inf'))
        mixed5b_pool = self.mixed5b_pool(mixed5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed5b_1x1 = self.mixed5b_1x1(mixed5b_1x1_pre_relu_conv)
        mixed5b_3x3_bottleneck = self.mixed5b_3x3_bottleneck(mixed5b_3x3_bottleneck_pre_relu_conv)
        mixed5b_5x5_bottleneck = self.mixed5b_5x5_bottleneck(mixed5b_5x5_bottleneck_pre_relu_conv)
        mixed5b_pool_reduce_pre_relu_conv = self.mixed5b_pool_reduce_pre_relu_conv(mixed5b_pool)
        mixed5b_3x3_pre_relu_conv_pad = F.pad(mixed5b_3x3_bottleneck, (1, 1, 1, 1))
        mixed5b_3x3_pre_relu_conv = self.mixed5b_3x3_pre_relu_conv(mixed5b_3x3_pre_relu_conv_pad)
        mixed5b_5x5_pre_relu_conv_pad = F.pad(mixed5b_5x5_bottleneck, (2, 2, 2, 2))
        mixed5b_5x5_pre_relu_conv = self.mixed5b_5x5_pre_relu_conv(mixed5b_5x5_pre_relu_conv_pad)
        mixed5b_pool_reduce = self.mixed5b_pool_reduce(mixed5b_pool_reduce_pre_relu_conv)
        mixed5b_3x3 = self.mixed5b_3x3(mixed5b_3x3_pre_relu_conv)
        mixed5b_5x5 = self.mixed5b_5x5(mixed5b_5x5_pre_relu_conv)
        mixed5b = self.mixed5b((mixed5b_1x1, mixed5b_3x3, mixed5b_5x5, mixed5b_pool_reduce), 1)
        avgpool0 = F.avg_pool2d(mixed5b, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        avgpool0_reshape = torch.reshape(input=avgpool0, shape=(-1, 1024))
        softmax2_pre_activation_matmul = self.softmax2_pre_activation_matmul(avgpool0_reshape)
        softmax2 = self.softmax2(softmax2_pre_activation_matmul)
        return softmax2


class InceptionV3Keras(nn.Module):

    def __init__(self):
        super(InceptionV3Keras, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_1 = nn.BatchNorm2d(num_features=32, eps=0.0010000000475, momentum=0.0)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_2 = nn.BatchNorm2d(num_features=32, eps=0.0010000000475, momentum=0.0)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_3 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_4 = nn.BatchNorm2d(num_features=80, eps=0.0010000000475, momentum=0.0)
        self.conv2d_5 = nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_5 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_9 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_7 = nn.Conv2d(in_channels=192, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_6 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_9 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_7 = nn.BatchNorm2d(num_features=48, eps=0.0010000000475, momentum=0.0)
        self.conv2d_12 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_6 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_12 = nn.BatchNorm2d(num_features=32, eps=0.0010000000475, momentum=0.0)
        self.conv2d_10 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_8 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_10 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_8 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_11 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_11 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.conv2d_16 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_14 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_13 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_16 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_14 = nn.BatchNorm2d(num_features=48, eps=0.0010000000475, momentum=0.0)
        self.conv2d_19 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_13 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_19 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_17 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_15 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_17 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_15 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_18 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_18 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.conv2d_23 = nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_21 = nn.Conv2d(in_channels=288, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_20 = nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_23 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_21 = nn.BatchNorm2d(num_features=48, eps=0.0010000000475, momentum=0.0)
        self.conv2d_26 = nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_20 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_26 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_24 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_22 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_24 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_22 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.conv2d_25 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_25 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.conv2d_28 = nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_27 = nn.Conv2d(in_channels=288, out_channels=384, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_28 = nn.BatchNorm2d(num_features=64, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_27 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_29 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_29 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.conv2d_30 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_30 = nn.BatchNorm2d(num_features=96, eps=0.0010000000475, momentum=0.0)
        self.conv2d_35 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_32 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_31 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_35 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_32 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.conv2d_40 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_31 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_40 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_36 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_33 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_36 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_33 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.conv2d_37 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.conv2d_34 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_37 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_34 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_38 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_38 = nn.BatchNorm2d(num_features=128, eps=0.0010000000475, momentum=0.0)
        self.conv2d_39 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_39 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_45 = nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_42 = nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_41 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_45 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_42 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_50 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_41 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_50 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_46 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_43 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_46 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_43 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_47 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.conv2d_44 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_47 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_44 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_48 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_48 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_49 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_49 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_55 = nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_52 = nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_51 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_55 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_52 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_60 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_51 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_60 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_56 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_53 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_56 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_53 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_57 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.conv2d_54 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_57 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_54 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_58 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_58 = nn.BatchNorm2d(num_features=160, eps=0.0010000000475, momentum=0.0)
        self.conv2d_59 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_59 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_65 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_62 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_61 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_65 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_62 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_70 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_61 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_70 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_66 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_63 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_66 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_63 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_67 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.conv2d_64 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_67 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_64 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_68 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_68 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_69 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_69 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_73 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_71 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_73 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_71 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_74 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        self.conv2d_72 = nn.Conv2d(in_channels=192, out_channels=320, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_74 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_72 = nn.BatchNorm2d(num_features=320, eps=0.0010000000475, momentum=0.0)
        self.conv2d_75 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_75 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_76 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_76 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_81 = nn.Conv2d(in_channels=1280, out_channels=448, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_78 = nn.Conv2d(in_channels=1280, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_77 = nn.Conv2d(in_channels=1280, out_channels=320, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_81 = nn.BatchNorm2d(num_features=448, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_78 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_85 = nn.Conv2d(in_channels=1280, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_77 = nn.BatchNorm2d(num_features=320, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_85 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_82 = nn.Conv2d(in_channels=448, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_79 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_80 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_82 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_79 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_80 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_83 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_84 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_83 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_84 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_90 = nn.Conv2d(in_channels=2048, out_channels=448, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_87 = nn.Conv2d(in_channels=2048, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2d_86 = nn.Conv2d(in_channels=2048, out_channels=320, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_90 = nn.BatchNorm2d(num_features=448, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_87 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_94 = nn.Conv2d(in_channels=2048, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_86 = nn.BatchNorm2d(num_features=320, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_94 = nn.BatchNorm2d(num_features=192, eps=0.0010000000475, momentum=0.0)
        self.conv2d_91 = nn.Conv2d(in_channels=448, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_88 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_89 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_91 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_88 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_89 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.conv2d_92 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_93 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), groups=1, bias=False)
        self.batch_normalization_92 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.batch_normalization_93 = nn.BatchNorm2d(num_features=384, eps=0.0010000000475, momentum=0.0)
        self.predictions = nn.Linear(in_features=2048, out_features=1000, bias=True)

    def add_layers(self):
        self.activation_1 = helper_layers.ReluLayer()
        self.activation_2 = helper_layers.ReluLayer()
        self.activation_3 = helper_layers.ReluLayer()
        self.max_pooling2d_1 = helper_layers.MaxPool2dLayer()
        self.activation_4 = helper_layers.ReluLayer()
        self.activation_5 = helper_layers.ReluLayer()
        self.max_pooling2d_2 = helper_layers.MaxPool2dLayer()
        self.average_pooling2d_1 = helper_layers.AVGPoolLayer()
        self.activation_9 = helper_layers.ReluLayer()
        self.activation_7 = helper_layers.ReluLayer()
        self.activation_6 = helper_layers.ReluLayer()
        self.activation_12 = helper_layers.ReluLayer()
        self.activation_10 = helper_layers.ReluLayer()
        self.activation_8 = helper_layers.ReluLayer()
        self.activation_11 = helper_layers.ReluLayer()
        self.mixed0 = helper_layers.CatLayer()
        self.average_pooling2d_2 = helper_layers.AVGPoolLayer()
        self.activation_16 = helper_layers.ReluLayer()
        self.activation_14 = helper_layers.ReluLayer()
        self.activation_13 = helper_layers.ReluLayer()
        self.activation_19 = helper_layers.ReluLayer()
        self.activation_17 = helper_layers.ReluLayer()
        self.activation_15 = helper_layers.ReluLayer()
        self.activation_18 = helper_layers.ReluLayer()
        self.mixed1 = helper_layers.CatLayer()
        self.average_pooling2d_3 = helper_layers.AVGPoolLayer()
        self.activation_23 = helper_layers.ReluLayer()
        self.activation_21 = helper_layers.ReluLayer()
        self.activation_20 = helper_layers.ReluLayer()
        self.activation_26 = helper_layers.ReluLayer()
        self.activation_24 = helper_layers.ReluLayer()
        self.activation_22 = helper_layers.ReluLayer()
        self.activation_25 = helper_layers.ReluLayer()
        self.mixed2 = helper_layers.CatLayer()
        self.max_pooling2d_3 = helper_layers.MaxPool2dLayer()
        self.activation_28 = helper_layers.ReluLayer()
        self.activation_27 = helper_layers.ReluLayer()
        self.activation_29 = helper_layers.ReluLayer()
        self.activation_30 = helper_layers.ReluLayer()
        self.mixed3 = helper_layers.CatLayer()
        self.average_pooling2d_4 = helper_layers.AVGPoolLayer()
        self.activation_35 = helper_layers.ReluLayer()
        self.activation_32 = helper_layers.ReluLayer()
        self.activation_31 = helper_layers.ReluLayer()
        self.activation_40 = helper_layers.ReluLayer()
        self.activation_36 = helper_layers.ReluLayer()
        self.activation_33 = helper_layers.ReluLayer()
        self.activation_37 = helper_layers.ReluLayer()
        self.activation_34 = helper_layers.ReluLayer()
        self.activation_38 = helper_layers.ReluLayer()
        self.activation_39 = helper_layers.ReluLayer()
        self.mixed4 = helper_layers.CatLayer()
        self.average_pooling2d_5 = helper_layers.AVGPoolLayer()
        self.activation_45 = helper_layers.ReluLayer()
        self.activation_42 = helper_layers.ReluLayer()
        self.activation_41 = helper_layers.ReluLayer()
        self.activation_50 = helper_layers.ReluLayer()
        self.activation_46 = helper_layers.ReluLayer()
        self.activation_43 = helper_layers.ReluLayer()
        self.activation_47 = helper_layers.ReluLayer()
        self.activation_44 = helper_layers.ReluLayer()
        self.activation_48 = helper_layers.ReluLayer()
        self.activation_49 = helper_layers.ReluLayer()
        self.mixed5 = helper_layers.CatLayer()
        self.average_pooling2d_6 = helper_layers.AVGPoolLayer()
        self.activation_55 = helper_layers.ReluLayer()
        self.activation_52 = helper_layers.ReluLayer()
        self.activation_51 = helper_layers.ReluLayer()
        self.activation_60 = helper_layers.ReluLayer()
        self.activation_56 = helper_layers.ReluLayer()
        self.activation_53 = helper_layers.ReluLayer()
        self.activation_57 = helper_layers.ReluLayer()
        self.activation_54 = helper_layers.ReluLayer()
        self.activation_58 = helper_layers.ReluLayer()
        self.activation_59 = helper_layers.ReluLayer()
        self.mixed6 = helper_layers.CatLayer()
        self.average_pooling2d_7 = helper_layers.AVGPoolLayer()
        self.activation_65 = helper_layers.ReluLayer()
        self.activation_62 = helper_layers.ReluLayer()
        self.activation_61 = helper_layers.ReluLayer()
        self.activation_70 = helper_layers.ReluLayer()
        self.activation_66 = helper_layers.ReluLayer()
        self.activation_63 = helper_layers.ReluLayer()
        self.activation_67 = helper_layers.ReluLayer()
        self.activation_64 = helper_layers.ReluLayer()
        self.activation_68 = helper_layers.ReluLayer()
        self.activation_69 = helper_layers.ReluLayer()
        self.mixed7 = helper_layers.CatLayer()
        self.max_pooling2d_4 = helper_layers.MaxPool2dLayer()
        self.activation_73 = helper_layers.ReluLayer()
        self.activation_71 = helper_layers.ReluLayer()
        self.activation_74 = helper_layers.ReluLayer()
        self.activation_72 = helper_layers.ReluLayer()
        self.activation_75 = helper_layers.ReluLayer()
        self.activation_76 = helper_layers.ReluLayer()
        self.mixed8 = helper_layers.CatLayer()
        self.average_pooling2d_8 = helper_layers.AVGPoolLayer()
        self.activation_81 = helper_layers.ReluLayer()
        self.activation_78 = helper_layers.ReluLayer()
        self.activation_77 = helper_layers.ReluLayer()
        self.activation_85 = helper_layers.ReluLayer()
        self.activation_82 = helper_layers.ReluLayer()
        self.activation_79 = helper_layers.ReluLayer()
        self.activation_80 = helper_layers.ReluLayer()
        self.mixed9_0 = helper_layers.CatLayer()
        self.activation_83 = helper_layers.ReluLayer()
        self.activation_84 = helper_layers.ReluLayer()
        self.concatenate_1 = helper_layers.CatLayer()
        self.mixed9 = helper_layers.CatLayer()
        self.average_pooling2d_9 = helper_layers.AVGPoolLayer()
        self.activation_90 = helper_layers.ReluLayer()
        self.activation_87 = helper_layers.ReluLayer()
        self.activation_86 = helper_layers.ReluLayer()
        self.activation_94 = helper_layers.ReluLayer()
        self.activation_91 = helper_layers.ReluLayer()
        self.activation_88 = helper_layers.ReluLayer()
        self.activation_89 = helper_layers.ReluLayer()
        self.mixed9_1 = helper_layers.CatLayer()
        self.activation_92 = helper_layers.ReluLayer()
        self.activation_93 = helper_layers.ReluLayer()
        self.concatenate_2 = helper_layers.CatLayer()
        self.mixed10 = helper_layers.CatLayer()
        self.avg_pool = helper_layers.AVGPoolLayer()
        self.predictions_activation = helper_layers.SoftMaxLayer()

    def forward(self, x):
        conv2d_1 = self.conv2d_1(x)
        batch_normalization_1 = self.batch_normalization_1(conv2d_1)
        activation_1 = self.activation_1(batch_normalization_1)
        conv2d_2 = self.conv2d_2(activation_1)
        batch_normalization_2 = self.batch_normalization_2(conv2d_2)
        activation_2 = self.activation_2(batch_normalization_2)
        conv2d_3_pad = F.pad(activation_2, (1, 1, 1, 1))
        conv2d_3 = self.conv2d_3(conv2d_3_pad)
        batch_normalization_3 = self.batch_normalization_3(conv2d_3)
        activation_3 = self.activation_3(batch_normalization_3)
        max_pooling2d_1 = self.max_pooling2d_1(activation_3, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_4 = self.conv2d_4(max_pooling2d_1)
        batch_normalization_4 = self.batch_normalization_4(conv2d_4)
        activation_4 = self.activation_4(batch_normalization_4)
        conv2d_5 = self.conv2d_5(activation_4)
        batch_normalization_5 = self.batch_normalization_5(conv2d_5)
        activation_5 = self.activation_5(batch_normalization_5)
        max_pooling2d_2 = self.max_pooling2d_2(activation_5, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_9 = self.conv2d_9(max_pooling2d_2)
        conv2d_7 = self.conv2d_7(max_pooling2d_2)
        average_pooling2d_1 = self.average_pooling2d_1(max_pooling2d_2, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_6 = self.conv2d_6(max_pooling2d_2)
        batch_normalization_9 = self.batch_normalization_9(conv2d_9)
        batch_normalization_7 = self.batch_normalization_7(conv2d_7)
        conv2d_12 = self.conv2d_12(average_pooling2d_1)
        batch_normalization_6 = self.batch_normalization_6(conv2d_6)
        activation_9 = self.activation_9(batch_normalization_9)
        activation_7 = self.activation_7(batch_normalization_7)
        batch_normalization_12 = self.batch_normalization_12(conv2d_12)
        activation_6 = self.activation_6(batch_normalization_6)
        conv2d_10_pad = F.pad(activation_9, (1, 1, 1, 1))
        conv2d_10 = self.conv2d_10(conv2d_10_pad)
        conv2d_8_pad = F.pad(activation_7, (2, 2, 2, 2))
        conv2d_8 = self.conv2d_8(conv2d_8_pad)
        activation_12 = self.activation_12(batch_normalization_12)
        batch_normalization_10 = self.batch_normalization_10(conv2d_10)
        batch_normalization_8 = self.batch_normalization_8(conv2d_8)
        activation_10 = self.activation_10(batch_normalization_10)
        activation_8 = self.activation_8(batch_normalization_8)
        conv2d_11_pad = F.pad(activation_10, (1, 1, 1, 1))
        conv2d_11 = self.conv2d_11(conv2d_11_pad)
        batch_normalization_11 = self.batch_normalization_11(conv2d_11)
        activation_11 = self.activation_11(batch_normalization_11)
        mixed0 = self.mixed0((activation_6, activation_8, activation_11, activation_12), 1)
        conv2d_16 = self.conv2d_16(mixed0)
        conv2d_14 = self.conv2d_14(mixed0)
        average_pooling2d_2 = self.average_pooling2d_2(mixed0, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_13 = self.conv2d_13(mixed0)
        batch_normalization_16 = self.batch_normalization_16(conv2d_16)
        batch_normalization_14 = self.batch_normalization_14(conv2d_14)
        conv2d_19 = self.conv2d_19(average_pooling2d_2)
        batch_normalization_13 = self.batch_normalization_13(conv2d_13)
        activation_16 = self.activation_16(batch_normalization_16)
        activation_14 = self.activation_14(batch_normalization_14)
        batch_normalization_19 = self.batch_normalization_19(conv2d_19)
        activation_13 = self.activation_13(batch_normalization_13)
        conv2d_17_pad = F.pad(activation_16, (1, 1, 1, 1))
        conv2d_17 = self.conv2d_17(conv2d_17_pad)
        conv2d_15_pad = F.pad(activation_14, (2, 2, 2, 2))
        conv2d_15 = self.conv2d_15(conv2d_15_pad)
        activation_19 = self.activation_19(batch_normalization_19)
        batch_normalization_17 = self.batch_normalization_17(conv2d_17)
        batch_normalization_15 = self.batch_normalization_15(conv2d_15)
        activation_17 = self.activation_17(batch_normalization_17)
        activation_15 = self.activation_15(batch_normalization_15)
        conv2d_18_pad = F.pad(activation_17, (1, 1, 1, 1))
        conv2d_18 = self.conv2d_18(conv2d_18_pad)
        batch_normalization_18 = self.batch_normalization_18(conv2d_18)
        activation_18 = self.activation_18(batch_normalization_18)
        mixed1 = self.mixed1((activation_13, activation_15, activation_18, activation_19), 1)
        conv2d_23 = self.conv2d_23(mixed1)
        conv2d_21 = self.conv2d_21(mixed1)
        average_pooling2d_3 = self.average_pooling2d_3(mixed1, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_20 = self.conv2d_20(mixed1)
        batch_normalization_23 = self.batch_normalization_23(conv2d_23)
        batch_normalization_21 = self.batch_normalization_21(conv2d_21)
        conv2d_26 = self.conv2d_26(average_pooling2d_3)
        batch_normalization_20 = self.batch_normalization_20(conv2d_20)
        activation_23 = self.activation_23(batch_normalization_23)
        activation_21 = self.activation_21(batch_normalization_21)
        batch_normalization_26 = self.batch_normalization_26(conv2d_26)
        activation_20 = self.activation_20(batch_normalization_20)
        conv2d_24_pad = F.pad(activation_23, (1, 1, 1, 1))
        conv2d_24 = self.conv2d_24(conv2d_24_pad)
        conv2d_22_pad = F.pad(activation_21, (2, 2, 2, 2))
        conv2d_22 = self.conv2d_22(conv2d_22_pad)
        activation_26 = self.activation_26(batch_normalization_26)
        batch_normalization_24 = self.batch_normalization_24(conv2d_24)
        batch_normalization_22 = self.batch_normalization_22(conv2d_22)
        activation_24 = self.activation_24(batch_normalization_24)
        activation_22 = self.activation_22(batch_normalization_22)
        conv2d_25_pad = F.pad(activation_24, (1, 1, 1, 1))
        conv2d_25 = self.conv2d_25(conv2d_25_pad)
        batch_normalization_25 = self.batch_normalization_25(conv2d_25)
        activation_25 = self.activation_25(batch_normalization_25)
        mixed2 = self.mixed2((activation_20, activation_22, activation_25, activation_26), 1)
        conv2d_28 = self.conv2d_28(mixed2)
        conv2d_27 = self.conv2d_27(mixed2)
        max_pooling2d_3 = self.max_pooling2d_3(mixed2, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_28 = self.batch_normalization_28(conv2d_28)
        batch_normalization_27 = self.batch_normalization_27(conv2d_27)
        activation_28 = self.activation_28(batch_normalization_28)
        activation_27 = self.activation_27(batch_normalization_27)
        conv2d_29_pad = F.pad(activation_28, (1, 1, 1, 1))
        conv2d_29 = self.conv2d_29(conv2d_29_pad)
        batch_normalization_29 = self.batch_normalization_29(conv2d_29)
        activation_29 = self.activation_29(batch_normalization_29)
        conv2d_30 = self.conv2d_30(activation_29)
        batch_normalization_30 = self.batch_normalization_30(conv2d_30)
        activation_30 = self.activation_30(batch_normalization_30)
        mixed3 = self.mixed3((activation_27, activation_30, max_pooling2d_3), 1)
        conv2d_35 = self.conv2d_35(mixed3)
        conv2d_32 = self.conv2d_32(mixed3)
        average_pooling2d_4 = self.average_pooling2d_4(mixed3, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_31 = self.conv2d_31(mixed3)
        batch_normalization_35 = self.batch_normalization_35(conv2d_35)
        batch_normalization_32 = self.batch_normalization_32(conv2d_32)
        conv2d_40 = self.conv2d_40(average_pooling2d_4)
        batch_normalization_31 = self.batch_normalization_31(conv2d_31)
        activation_35 = self.activation_35(batch_normalization_35)
        activation_32 = self.activation_32(batch_normalization_32)
        batch_normalization_40 = self.batch_normalization_40(conv2d_40)
        activation_31 = self.activation_31(batch_normalization_31)
        conv2d_36_pad = F.pad(activation_35, (0, 0, 3, 3))
        conv2d_36 = self.conv2d_36(conv2d_36_pad)
        conv2d_33_pad = F.pad(activation_32, (3, 3, 0, 0))
        conv2d_33 = self.conv2d_33(conv2d_33_pad)
        activation_40 = self.activation_40(batch_normalization_40)
        batch_normalization_36 = self.batch_normalization_36(conv2d_36)
        batch_normalization_33 = self.batch_normalization_33(conv2d_33)
        activation_36 = self.activation_36(batch_normalization_36)
        activation_33 = self.activation_33(batch_normalization_33)
        conv2d_37_pad = F.pad(activation_36, (3, 3, 0, 0))
        conv2d_37 = self.conv2d_37(conv2d_37_pad)
        conv2d_34_pad = F.pad(activation_33, (0, 0, 3, 3))
        conv2d_34 = self.conv2d_34(conv2d_34_pad)
        batch_normalization_37 = self.batch_normalization_37(conv2d_37)
        batch_normalization_34 = self.batch_normalization_34(conv2d_34)
        activation_37 = self.activation_37(batch_normalization_37)
        activation_34 = self.activation_34(batch_normalization_34)
        conv2d_38_pad = F.pad(activation_37, (0, 0, 3, 3))
        conv2d_38 = self.conv2d_38(conv2d_38_pad)
        batch_normalization_38 = self.batch_normalization_38(conv2d_38)
        activation_38 = self.activation_38(batch_normalization_38)
        conv2d_39_pad = F.pad(activation_38, (3, 3, 0, 0))
        conv2d_39 = self.conv2d_39(conv2d_39_pad)
        batch_normalization_39 = self.batch_normalization_39(conv2d_39)
        activation_39 = self.activation_39(batch_normalization_39)
        mixed4 = self.mixed4((activation_31, activation_34, activation_39, activation_40), 1)
        conv2d_45 = self.conv2d_45(mixed4)
        conv2d_42 = self.conv2d_42(mixed4)
        average_pooling2d_5 = self.average_pooling2d_5(mixed4, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_41 = self.conv2d_41(mixed4)
        batch_normalization_45 = self.batch_normalization_45(conv2d_45)
        batch_normalization_42 = self.batch_normalization_42(conv2d_42)
        conv2d_50 = self.conv2d_50(average_pooling2d_5)
        batch_normalization_41 = self.batch_normalization_41(conv2d_41)
        activation_45 = self.activation_45(batch_normalization_45)
        activation_42 = self.activation_42(batch_normalization_42)
        batch_normalization_50 = self.batch_normalization_50(conv2d_50)
        activation_41 = self.activation_41(batch_normalization_41)
        conv2d_46_pad = F.pad(activation_45, (0, 0, 3, 3))
        conv2d_46 = self.conv2d_46(conv2d_46_pad)
        conv2d_43_pad = F.pad(activation_42, (3, 3, 0, 0))
        conv2d_43 = self.conv2d_43(conv2d_43_pad)
        activation_50 = self.activation_50(batch_normalization_50)
        batch_normalization_46 = self.batch_normalization_46(conv2d_46)
        batch_normalization_43 = self.batch_normalization_43(conv2d_43)
        activation_46 = self.activation_46(batch_normalization_46)
        activation_43 = self.activation_43(batch_normalization_43)
        conv2d_47_pad = F.pad(activation_46, (3, 3, 0, 0))
        conv2d_47 = self.conv2d_47(conv2d_47_pad)
        conv2d_44_pad = F.pad(activation_43, (0, 0, 3, 3))
        conv2d_44 = self.conv2d_44(conv2d_44_pad)
        batch_normalization_47 = self.batch_normalization_47(conv2d_47)
        batch_normalization_44 = self.batch_normalization_44(conv2d_44)
        activation_47 = self.activation_47(batch_normalization_47)
        activation_44 = self.activation_44(batch_normalization_44)
        conv2d_48_pad = F.pad(activation_47, (0, 0, 3, 3))
        conv2d_48 = self.conv2d_48(conv2d_48_pad)
        batch_normalization_48 = self.batch_normalization_48(conv2d_48)
        activation_48 = self.activation_48(batch_normalization_48)
        conv2d_49_pad = F.pad(activation_48, (3, 3, 0, 0))
        conv2d_49 = self.conv2d_49(conv2d_49_pad)
        batch_normalization_49 = self.batch_normalization_49(conv2d_49)
        activation_49 = self.activation_49(batch_normalization_49)
        mixed5 = self.mixed5((activation_41, activation_44, activation_49, activation_50), 1)
        conv2d_55 = self.conv2d_55(mixed5)
        conv2d_52 = self.conv2d_52(mixed5)
        average_pooling2d_6 = self.average_pooling2d_6(mixed5, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_51 = self.conv2d_51(mixed5)
        batch_normalization_55 = self.batch_normalization_55(conv2d_55)
        batch_normalization_52 = self.batch_normalization_52(conv2d_52)
        conv2d_60 = self.conv2d_60(average_pooling2d_6)
        batch_normalization_51 = self.batch_normalization_51(conv2d_51)
        activation_55 = self.activation_55(batch_normalization_55)
        activation_52 = self.activation_52(batch_normalization_52)
        batch_normalization_60 = self.batch_normalization_60(conv2d_60)
        activation_51 = self.activation_51(batch_normalization_51)
        conv2d_56_pad = F.pad(activation_55, (0, 0, 3, 3))
        conv2d_56 = self.conv2d_56(conv2d_56_pad)
        conv2d_53_pad = F.pad(activation_52, (3, 3, 0, 0))
        conv2d_53 = self.conv2d_53(conv2d_53_pad)
        activation_60 = self.activation_60(batch_normalization_60)
        batch_normalization_56 = self.batch_normalization_56(conv2d_56)
        batch_normalization_53 = self.batch_normalization_53(conv2d_53)
        activation_56 = self.activation_56(batch_normalization_56)
        activation_53 = self.activation_53(batch_normalization_53)
        conv2d_57_pad = F.pad(activation_56, (3, 3, 0, 0))
        conv2d_57 = self.conv2d_57(conv2d_57_pad)
        conv2d_54_pad = F.pad(activation_53, (0, 0, 3, 3))
        conv2d_54 = self.conv2d_54(conv2d_54_pad)
        batch_normalization_57 = self.batch_normalization_57(conv2d_57)
        batch_normalization_54 = self.batch_normalization_54(conv2d_54)
        activation_57 = self.activation_57(batch_normalization_57)
        activation_54 = self.activation_54(batch_normalization_54)
        conv2d_58_pad = F.pad(activation_57, (0, 0, 3, 3))
        conv2d_58 = self.conv2d_58(conv2d_58_pad)
        batch_normalization_58 = self.batch_normalization_58(conv2d_58)
        activation_58 = self.activation_58(batch_normalization_58)
        conv2d_59_pad = F.pad(activation_58, (3, 3, 0, 0))
        conv2d_59 = self.conv2d_59(conv2d_59_pad)
        batch_normalization_59 = self.batch_normalization_59(conv2d_59)
        activation_59 = self.activation_59(batch_normalization_59)
        mixed6 = self.mixed6((activation_51, activation_54, activation_59, activation_60), 1)
        conv2d_65 = self.conv2d_65(mixed6)
        conv2d_62 = self.conv2d_62(mixed6)
        average_pooling2d_7 = self.average_pooling2d_7(mixed6, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_61 = self.conv2d_61(mixed6)
        batch_normalization_65 = self.batch_normalization_65(conv2d_65)
        batch_normalization_62 = self.batch_normalization_62(conv2d_62)
        conv2d_70 = self.conv2d_70(average_pooling2d_7)
        batch_normalization_61 = self.batch_normalization_61(conv2d_61)
        activation_65 = self.activation_65(batch_normalization_65)
        activation_62 = self.activation_62(batch_normalization_62)
        batch_normalization_70 = self.batch_normalization_70(conv2d_70)
        activation_61 = self.activation_61(batch_normalization_61)
        conv2d_66_pad = F.pad(activation_65, (0, 0, 3, 3))
        conv2d_66 = self.conv2d_66(conv2d_66_pad)
        conv2d_63_pad = F.pad(activation_62, (3, 3, 0, 0))
        conv2d_63 = self.conv2d_63(conv2d_63_pad)
        activation_70 = self.activation_70(batch_normalization_70)
        batch_normalization_66 = self.batch_normalization_66(conv2d_66)
        batch_normalization_63 = self.batch_normalization_63(conv2d_63)
        activation_66 = self.activation_66(batch_normalization_66)
        activation_63 = self.activation_63(batch_normalization_63)
        conv2d_67_pad = F.pad(activation_66, (3, 3, 0, 0))
        conv2d_67 = self.conv2d_67(conv2d_67_pad)
        conv2d_64_pad = F.pad(activation_63, (0, 0, 3, 3))
        conv2d_64 = self.conv2d_64(conv2d_64_pad)
        batch_normalization_67 = self.batch_normalization_67(conv2d_67)
        batch_normalization_64 = self.batch_normalization_64(conv2d_64)
        activation_67 = self.activation_67(batch_normalization_67)
        activation_64 = self.activation_64(batch_normalization_64)
        conv2d_68_pad = F.pad(activation_67, (0, 0, 3, 3))
        conv2d_68 = self.conv2d_68(conv2d_68_pad)
        batch_normalization_68 = self.batch_normalization_68(conv2d_68)
        activation_68 = self.activation_68(batch_normalization_68)
        conv2d_69_pad = F.pad(activation_68, (3, 3, 0, 0))
        conv2d_69 = self.conv2d_69(conv2d_69_pad)
        batch_normalization_69 = self.batch_normalization_69(conv2d_69)
        activation_69 = self.activation_69(batch_normalization_69)
        mixed7 = self.mixed7((activation_61, activation_64, activation_69, activation_70), 1)
        conv2d_73 = self.conv2d_73(mixed7)
        conv2d_71 = self.conv2d_71(mixed7)
        max_pooling2d_4 = self.max_pooling2d_4(mixed7, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_73 = self.batch_normalization_73(conv2d_73)
        batch_normalization_71 = self.batch_normalization_71(conv2d_71)
        activation_73 = self.activation_73(batch_normalization_73)
        activation_71 = self.activation_71(batch_normalization_71)
        conv2d_74_pad = F.pad(activation_73, (3, 3, 0, 0))
        conv2d_74 = self.conv2d_74(conv2d_74_pad)
        conv2d_72 = self.conv2d_72(activation_71)
        batch_normalization_74 = self.batch_normalization_74(conv2d_74)
        batch_normalization_72 = self.batch_normalization_72(conv2d_72)
        activation_74 = self.activation_74(batch_normalization_74)
        activation_72 = self.activation_72(batch_normalization_72)
        conv2d_75_pad = F.pad(activation_74, (0, 0, 3, 3))
        conv2d_75 = self.conv2d_75(conv2d_75_pad)
        batch_normalization_75 = self.batch_normalization_75(conv2d_75)
        activation_75 = self.activation_75(batch_normalization_75)
        conv2d_76 = self.conv2d_76(activation_75)
        batch_normalization_76 = self.batch_normalization_76(conv2d_76)
        activation_76 = self.activation_76(batch_normalization_76)
        mixed8 = self.mixed8((activation_72, activation_76, max_pooling2d_4), 1)
        conv2d_81 = self.conv2d_81(mixed8)
        conv2d_78 = self.conv2d_78(mixed8)
        average_pooling2d_8 = self.average_pooling2d_8(mixed8, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_77 = self.conv2d_77(mixed8)
        batch_normalization_81 = self.batch_normalization_81(conv2d_81)
        batch_normalization_78 = self.batch_normalization_78(conv2d_78)
        conv2d_85 = self.conv2d_85(average_pooling2d_8)
        batch_normalization_77 = self.batch_normalization_77(conv2d_77)
        activation_81 = self.activation_81(batch_normalization_81)
        activation_78 = self.activation_78(batch_normalization_78)
        batch_normalization_85 = self.batch_normalization_85(conv2d_85)
        activation_77 = self.activation_77(batch_normalization_77)
        conv2d_82_pad = F.pad(activation_81, (1, 1, 1, 1))
        conv2d_82 = self.conv2d_82(conv2d_82_pad)
        conv2d_79_pad = F.pad(activation_78, (1, 1, 0, 0))
        conv2d_79 = self.conv2d_79(conv2d_79_pad)
        conv2d_80_pad = F.pad(activation_78, (0, 0, 1, 1))
        conv2d_80 = self.conv2d_80(conv2d_80_pad)
        activation_85 = self.activation_85(batch_normalization_85)
        batch_normalization_82 = self.batch_normalization_82(conv2d_82)
        batch_normalization_79 = self.batch_normalization_79(conv2d_79)
        batch_normalization_80 = self.batch_normalization_80(conv2d_80)
        activation_82 = self.activation_82(batch_normalization_82)
        activation_79 = self.activation_79(batch_normalization_79)
        activation_80 = self.activation_80(batch_normalization_80)
        conv2d_83_pad = F.pad(activation_82, (1, 1, 0, 0))
        conv2d_83 = self.conv2d_83(conv2d_83_pad)
        conv2d_84_pad = F.pad(activation_82, (0, 0, 1, 1))
        conv2d_84 = self.conv2d_84(conv2d_84_pad)
        mixed9_0 = self.mixed9_0((activation_79, activation_80), 1)
        batch_normalization_83 = self.batch_normalization_83(conv2d_83)
        batch_normalization_84 = self.batch_normalization_84(conv2d_84)
        activation_83 = self.activation_83(batch_normalization_83)
        activation_84 = self.activation_84(batch_normalization_84)
        concatenate_1 = self.concatenate_1((activation_83, activation_84), 1)
        mixed9 = self.mixed9((activation_77, mixed9_0, concatenate_1, activation_85), 1)
        conv2d_90 = self.conv2d_90(mixed9)
        conv2d_87 = self.conv2d_87(mixed9)
        average_pooling2d_9 = self.average_pooling2d_9(mixed9, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        conv2d_86 = self.conv2d_86(mixed9)
        batch_normalization_90 = self.batch_normalization_90(conv2d_90)
        batch_normalization_87 = self.batch_normalization_87(conv2d_87)
        conv2d_94 = self.conv2d_94(average_pooling2d_9)
        batch_normalization_86 = self.batch_normalization_86(conv2d_86)
        activation_90 = self.activation_90(batch_normalization_90)
        activation_87 = self.activation_87(batch_normalization_87)
        batch_normalization_94 = self.batch_normalization_94(conv2d_94)
        activation_86 = self.activation_86(batch_normalization_86)
        conv2d_91_pad = F.pad(activation_90, (1, 1, 1, 1))
        conv2d_91 = self.conv2d_91(conv2d_91_pad)
        conv2d_88_pad = F.pad(activation_87, (1, 1, 0, 0))
        conv2d_88 = self.conv2d_88(conv2d_88_pad)
        conv2d_89_pad = F.pad(activation_87, (0, 0, 1, 1))
        conv2d_89 = self.conv2d_89(conv2d_89_pad)
        activation_94 = self.activation_94(batch_normalization_94)
        batch_normalization_91 = self.batch_normalization_91(conv2d_91)
        batch_normalization_88 = self.batch_normalization_88(conv2d_88)
        batch_normalization_89 = self.batch_normalization_89(conv2d_89)
        activation_91 = self.activation_91(batch_normalization_91)
        activation_88 = self.activation_88(batch_normalization_88)
        activation_89 = self.activation_89(batch_normalization_89)
        conv2d_92_pad = F.pad(activation_91, (1, 1, 0, 0))
        conv2d_92 = self.conv2d_92(conv2d_92_pad)
        conv2d_93_pad = F.pad(activation_91, (0, 0, 1, 1))
        conv2d_93 = self.conv2d_93(conv2d_93_pad)
        mixed9_1 = self.mixed9_1((activation_88, activation_89), 1)
        batch_normalization_92 = self.batch_normalization_92(conv2d_92)
        batch_normalization_93 = self.batch_normalization_93(conv2d_93)
        activation_92 = self.activation_92(batch_normalization_92)
        activation_93 = self.activation_93(batch_normalization_93)
        concatenate_2 = self.concatenate_2((activation_92, activation_93), 1)
        mixed10 = self.mixed10((activation_86, mixed9_1, concatenate_2, activation_94), 1)
        avg_pool = self.avg_pool(input=mixed10, kernel_size=mixed10.size()[2:])
        avg_pool_flatten = avg_pool.view(avg_pool.size(0), -1)
        predictions = self.predictions(avg_pool_flatten)
        predictions_activation = self.predictions_activation(predictions)
        return predictions_activation


class ResNet_50_1by2_nsfw(nn.Module):

    def __init__(self):
        super(ResNet_50_1by2_nsfw, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block0_branch2a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_stage0_block0_proj_shortcut = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block0_branch2a = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_stage0_block0_proj_shortcut = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block0_branch2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block0_branch2b = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block0_branch2c = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block0_branch2c = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block1_branch2a = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block1_branch2a = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block1_branch2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block1_branch2b = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block1_branch2c = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block1_branch2c = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block2_branch2a = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block2_branch2a = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block2_branch2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block2_branch2b = nn.BatchNorm2d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage0_block2_branch2c = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage0_block2_branch2c = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block0_proj_shortcut = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.conv_stage1_block0_branch2a = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.bn_stage1_block0_proj_shortcut = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_stage1_block0_branch2a = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block0_branch2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block0_branch2b = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block0_branch2c = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block0_branch2c = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block1_branch2a = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block1_branch2a = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block1_branch2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block1_branch2b = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block1_branch2c = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block1_branch2c = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block2_branch2a = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block2_branch2a = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block2_branch2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block2_branch2b = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block2_branch2c = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block2_branch2c = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block3_branch2a = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block3_branch2a = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block3_branch2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block3_branch2b = nn.BatchNorm2d(num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage1_block3_branch2c = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage1_block3_branch2c = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block0_proj_shortcut = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.conv_stage2_block0_branch2a = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.bn_stage2_block0_proj_shortcut = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_stage2_block0_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block0_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block0_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block0_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block0_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block1_branch2a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block1_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block1_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block1_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block1_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block1_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block2_branch2a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block2_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block2_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block2_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block2_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block2_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block3_branch2a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block3_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block3_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block3_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block3_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block3_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block4_branch2a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block4_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block4_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block4_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block4_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block4_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block5_branch2a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block5_branch2a = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block5_branch2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block5_branch2b = nn.BatchNorm2d(num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage2_block5_branch2c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage2_block5_branch2c = nn.BatchNorm2d(num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block0_proj_shortcut = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.conv_stage3_block0_branch2a = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.bn_stage3_block0_proj_shortcut = nn.BatchNorm2d(num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_stage3_block0_branch2a = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block0_branch2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block0_branch2b = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block0_branch2c = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block0_branch2c = nn.BatchNorm2d(num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block1_branch2a = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block1_branch2a = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block1_branch2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block1_branch2b = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block1_branch2c = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block1_branch2c = nn.BatchNorm2d(num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block2_branch2a = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block2_branch2a = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block2_branch2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block2_branch2b = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_stage3_block2_branch2c = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_stage3_block2_branch2c = nn.BatchNorm2d(num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.fc_nsfw_1 = nn.Linear(in_features=1024, out_features=2, bias=True)

    def add_layers(self):
        self.eltwise_stage0_block0 = helper_layers.AdditionLayer()
        self.eltwise_stage0_block1 = helper_layers.AdditionLayer()
        self.eltwise_stage0_block2 = helper_layers.AdditionLayer()
        self.eltwise_stage1_block0 = helper_layers.AdditionLayer()
        self.eltwise_stage1_block1 = helper_layers.AdditionLayer()
        self.eltwise_stage1_block2 = helper_layers.AdditionLayer()
        self.eltwise_stage1_block3 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block0 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block1 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block2 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block3 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block4 = helper_layers.AdditionLayer()
        self.eltwise_stage2_block5 = helper_layers.AdditionLayer()
        self.eltwise_stage3_block0 = helper_layers.AdditionLayer()
        self.eltwise_stage3_block1 = helper_layers.AdditionLayer()
        self.eltwise_stage3_block2 = helper_layers.AdditionLayer()
        self.relu_1 = helper_layers.ReluLayer()
        self.relu_stage0_block0_branch2a = helper_layers.ReluLayer()
        self.relu_stage0_block0_branch2b = helper_layers.ReluLayer()
        self.relu_stage0_block0 = helper_layers.ReluLayer()
        self.relu_stage0_block1_branch2a = helper_layers.ReluLayer()
        self.relu_stage0_block1_branch2b = helper_layers.ReluLayer()
        self.relu_stage0_block1 = helper_layers.ReluLayer()
        self.relu_stage0_block2_branch2a = helper_layers.ReluLayer()
        self.relu_stage0_block2_branch2b = helper_layers.ReluLayer()
        self.relu_stage0_block2 = helper_layers.ReluLayer()
        self.relu_stage1_block0_branch2a = helper_layers.ReluLayer()
        self.relu_stage1_block0_branch2b = helper_layers.ReluLayer()
        self.relu_stage1_block0 = helper_layers.ReluLayer()
        self.relu_stage1_block1_branch2a = helper_layers.ReluLayer()
        self.relu_stage1_block1_branch2b = helper_layers.ReluLayer()
        self.relu_stage1_block1 = helper_layers.ReluLayer()
        self.relu_stage1_block2_branch2a = helper_layers.ReluLayer()
        self.relu_stage1_block2_branch2b = helper_layers.ReluLayer()
        self.relu_stage1_block2 = helper_layers.ReluLayer()
        self.relu_stage1_block3_branch2a = helper_layers.ReluLayer()
        self.relu_stage1_block3_branch2b = helper_layers.ReluLayer()
        self.relu_stage1_block3 = helper_layers.ReluLayer()
        self.relu_stage2_block0_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block0_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block0 = helper_layers.ReluLayer()
        self.relu_stage2_block1_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block1_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block1 = helper_layers.ReluLayer()
        self.relu_stage2_block2_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block2_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block2 = helper_layers.ReluLayer()
        self.relu_stage2_block3_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block3_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block3 = helper_layers.ReluLayer()
        self.relu_stage2_block4_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block4_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block4 = helper_layers.ReluLayer()
        self.relu_stage2_block5_branch2a = helper_layers.ReluLayer()
        self.relu_stage2_block5_branch2b = helper_layers.ReluLayer()
        self.relu_stage2_block5 = helper_layers.ReluLayer()
        self.relu_stage3_block0_branch2a = helper_layers.ReluLayer()
        self.relu_stage3_block0_branch2b = helper_layers.ReluLayer()
        self.relu_stage3_block0 = helper_layers.ReluLayer()
        self.relu_stage3_block1_branch2a = helper_layers.ReluLayer()
        self.relu_stage3_block1_branch2b = helper_layers.ReluLayer()
        self.relu_stage3_block1 = helper_layers.ReluLayer()
        self.relu_stage3_block2_branch2a = helper_layers.ReluLayer()
        self.relu_stage3_block2_branch2b = helper_layers.ReluLayer()
        self.relu_stage3_block2 = helper_layers.ReluLayer()
        self.conv_1_pad = helper_layers.PadLayer()
        self.pool1_pad = helper_layers.PadLayer()
        self.conv_stage0_block0_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage0_block1_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage0_block2_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage1_block0_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage1_block1_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage1_block2_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage1_block3_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block0_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block1_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block2_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block3_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block4_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage2_block5_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage3_block0_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage3_block1_branch2b_pad = helper_layers.PadLayer()
        self.conv_stage3_block2_branch2b_pad = helper_layers.PadLayer()
        self.pool1 = helper_layers.MaxPool2dLayer()
        self.prob = helper_layers.SoftMaxLayer()

    def forward(self, input):
        conv_1_pad = self.conv_1_pad(input, (3, 3, 3, 3))
        conv_1 = self.conv_1(conv_1_pad)
        bn_1 = self.bn_1(conv_1)
        relu_1 = self.relu_1(bn_1)
        pool1_pad = self.pool1_pad(relu_1, (0, 1, 0, 1), value=float('-inf'))
        pool1 = self.pool1(pool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv_stage0_block0_branch2a = self.conv_stage0_block0_branch2a(pool1)
        conv_stage0_block0_proj_shortcut = self.conv_stage0_block0_proj_shortcut(pool1)
        bn_stage0_block0_branch2a = self.bn_stage0_block0_branch2a(conv_stage0_block0_branch2a)
        bn_stage0_block0_proj_shortcut = self.bn_stage0_block0_proj_shortcut(conv_stage0_block0_proj_shortcut)
        relu_stage0_block0_branch2a = self.relu_stage0_block0_branch2a(bn_stage0_block0_branch2a)
        conv_stage0_block0_branch2b_pad = self.conv_stage0_block0_branch2b_pad(relu_stage0_block0_branch2a, (1, 1, 1, 1))
        conv_stage0_block0_branch2b = self.conv_stage0_block0_branch2b(conv_stage0_block0_branch2b_pad)
        bn_stage0_block0_branch2b = self.bn_stage0_block0_branch2b(conv_stage0_block0_branch2b)
        relu_stage0_block0_branch2b = self.relu_stage0_block0_branch2b(bn_stage0_block0_branch2b)
        conv_stage0_block0_branch2c = self.conv_stage0_block0_branch2c(relu_stage0_block0_branch2b)
        bn_stage0_block0_branch2c = self.bn_stage0_block0_branch2c(conv_stage0_block0_branch2c)
        eltwise_stage0_block0 = self.eltwise_stage0_block0(bn_stage0_block0_proj_shortcut, bn_stage0_block0_branch2c)
        relu_stage0_block0 = self.relu_stage0_block0(eltwise_stage0_block0)
        conv_stage0_block1_branch2a = self.conv_stage0_block1_branch2a(relu_stage0_block0)
        bn_stage0_block1_branch2a = self.bn_stage0_block1_branch2a(conv_stage0_block1_branch2a)
        relu_stage0_block1_branch2a = self.relu_stage0_block1_branch2a(bn_stage0_block1_branch2a)
        conv_stage0_block1_branch2b_pad = self.conv_stage0_block1_branch2b_pad(relu_stage0_block1_branch2a, (1, 1, 1, 1))
        conv_stage0_block1_branch2b = self.conv_stage0_block1_branch2b(conv_stage0_block1_branch2b_pad)
        bn_stage0_block1_branch2b = self.bn_stage0_block1_branch2b(conv_stage0_block1_branch2b)
        relu_stage0_block1_branch2b = self.relu_stage0_block1_branch2b(bn_stage0_block1_branch2b)
        conv_stage0_block1_branch2c = self.conv_stage0_block1_branch2c(relu_stage0_block1_branch2b)
        conv_stage0_block1_branch2c = self.conv_stage0_block1_branch2c(relu_stage0_block1_branch2b)
        bn_stage0_block1_branch2c = self.bn_stage0_block1_branch2c(conv_stage0_block1_branch2c)
        eltwise_stage0_block1 = self.eltwise_stage0_block1(relu_stage0_block0, bn_stage0_block1_branch2c)
        relu_stage0_block1 = self.relu_stage0_block1(eltwise_stage0_block1)
        conv_stage0_block2_branch2a = self.conv_stage0_block2_branch2a(relu_stage0_block1)
        bn_stage0_block2_branch2a = self.bn_stage0_block2_branch2a(conv_stage0_block2_branch2a)
        relu_stage0_block2_branch2a = self.relu_stage0_block2_branch2a(bn_stage0_block2_branch2a)
        conv_stage0_block2_branch2b_pad = self.conv_stage0_block2_branch2b_pad(relu_stage0_block2_branch2a, (1, 1, 1, 1))
        conv_stage0_block2_branch2b = self.conv_stage0_block2_branch2b(conv_stage0_block2_branch2b_pad)
        bn_stage0_block2_branch2b = self.bn_stage0_block2_branch2b(conv_stage0_block2_branch2b)
        relu_stage0_block2_branch2b = self.relu_stage0_block2_branch2b(bn_stage0_block2_branch2b)
        conv_stage0_block2_branch2c = self.conv_stage0_block2_branch2c(relu_stage0_block2_branch2b)
        bn_stage0_block2_branch2c = self.bn_stage0_block2_branch2c(conv_stage0_block2_branch2c)
        eltwise_stage0_block2 = self.eltwise_stage0_block2(relu_stage0_block1, bn_stage0_block2_branch2c)
        relu_stage0_block2 = self.relu_stage0_block2(eltwise_stage0_block2)
        conv_stage1_block0_proj_shortcut = self.conv_stage1_block0_proj_shortcut(relu_stage0_block2)
        conv_stage1_block0_branch2a = self.conv_stage1_block0_branch2a(relu_stage0_block2)
        bn_stage1_block0_proj_shortcut = self.bn_stage1_block0_proj_shortcut(conv_stage1_block0_proj_shortcut)
        bn_stage1_block0_branch2a = self.bn_stage1_block0_branch2a(conv_stage1_block0_branch2a)
        relu_stage1_block0_branch2a = self.relu_stage1_block0_branch2a(bn_stage1_block0_branch2a)
        conv_stage1_block0_branch2b_pad = self.conv_stage1_block0_branch2b_pad(relu_stage1_block0_branch2a, (1, 1, 1, 1))
        conv_stage1_block0_branch2b = self.conv_stage1_block0_branch2b(conv_stage1_block0_branch2b_pad)
        bn_stage1_block0_branch2b = self.bn_stage1_block0_branch2b(conv_stage1_block0_branch2b)
        relu_stage1_block0_branch2b = self.relu_stage1_block0_branch2b(bn_stage1_block0_branch2b)
        conv_stage1_block0_branch2c = self.conv_stage1_block0_branch2c(relu_stage1_block0_branch2b)
        bn_stage1_block0_branch2c = self.bn_stage1_block0_branch2c(conv_stage1_block0_branch2c)
        eltwise_stage1_block0 = self.eltwise_stage1_block0(bn_stage1_block0_proj_shortcut, bn_stage1_block0_branch2c)
        relu_stage1_block0 = self.relu_stage1_block0(eltwise_stage1_block0)
        conv_stage1_block1_branch2a = self.conv_stage1_block1_branch2a(relu_stage1_block0)
        bn_stage1_block1_branch2a = self.bn_stage1_block1_branch2a(conv_stage1_block1_branch2a)
        relu_stage1_block1_branch2a = self.relu_stage1_block1_branch2a(bn_stage1_block1_branch2a)
        conv_stage1_block1_branch2b_pad = self.conv_stage1_block1_branch2b_pad(relu_stage1_block1_branch2a, (1, 1, 1, 1))
        conv_stage1_block1_branch2b = self.conv_stage1_block1_branch2b(conv_stage1_block1_branch2b_pad)
        bn_stage1_block1_branch2b = self.bn_stage1_block1_branch2b(conv_stage1_block1_branch2b)
        relu_stage1_block1_branch2b = self.relu_stage1_block1_branch2b(bn_stage1_block1_branch2b)
        conv_stage1_block1_branch2c = self.conv_stage1_block1_branch2c(relu_stage1_block1_branch2b)
        bn_stage1_block1_branch2c = self.bn_stage1_block1_branch2c(conv_stage1_block1_branch2c)
        eltwise_stage1_block1 = relu_stage1_block0 + bn_stage1_block1_branch2c
        relu_stage1_block1 = self.relu_stage1_block1(eltwise_stage1_block1)
        conv_stage1_block2_branch2a = self.conv_stage1_block2_branch2a(relu_stage1_block1)
        bn_stage1_block2_branch2a = self.bn_stage1_block2_branch2a(conv_stage1_block2_branch2a)
        relu_stage1_block2_branch2a = self.relu_stage1_block2_branch2a(bn_stage1_block2_branch2a)
        conv_stage1_block2_branch2b_pad = self.conv_stage1_block2_branch2b_pad(relu_stage1_block2_branch2a, (1, 1, 1, 1))
        conv_stage1_block2_branch2b = self.conv_stage1_block2_branch2b(conv_stage1_block2_branch2b_pad)
        bn_stage1_block2_branch2b = self.bn_stage1_block2_branch2b(conv_stage1_block2_branch2b)
        relu_stage1_block2_branch2b = self.relu_stage1_block2_branch2b(bn_stage1_block2_branch2b)
        conv_stage1_block2_branch2c = self.conv_stage1_block2_branch2c(relu_stage1_block2_branch2b)
        bn_stage1_block2_branch2c = self.bn_stage1_block2_branch2c(conv_stage1_block2_branch2c)
        eltwise_stage1_block2 = self.eltwise_stage1_block2(relu_stage1_block1, bn_stage1_block2_branch2c)
        relu_stage1_block2 = self.relu_stage1_block2(eltwise_stage1_block2)
        conv_stage1_block3_branch2a = self.conv_stage1_block3_branch2a(relu_stage1_block2)
        bn_stage1_block3_branch2a = self.bn_stage1_block3_branch2a(conv_stage1_block3_branch2a)
        relu_stage1_block3_branch2a = self.relu_stage1_block3_branch2a(bn_stage1_block3_branch2a)
        conv_stage1_block3_branch2b_pad = self.conv_stage1_block3_branch2b_pad(relu_stage1_block3_branch2a, (1, 1, 1, 1))
        conv_stage1_block3_branch2b = self.conv_stage1_block3_branch2b(conv_stage1_block3_branch2b_pad)
        bn_stage1_block3_branch2b = self.bn_stage1_block3_branch2b(conv_stage1_block3_branch2b)
        relu_stage1_block3_branch2b = self.relu_stage1_block3_branch2b(bn_stage1_block3_branch2b)
        conv_stage1_block3_branch2c = self.conv_stage1_block3_branch2c(relu_stage1_block3_branch2b)
        bn_stage1_block3_branch2c = self.bn_stage1_block3_branch2c(conv_stage1_block3_branch2c)
        eltwise_stage1_block3 = self.eltwise_stage1_block3(relu_stage1_block2, bn_stage1_block3_branch2c)
        relu_stage1_block3 = self.relu_stage1_block3(eltwise_stage1_block3)
        conv_stage2_block0_proj_shortcut = self.conv_stage2_block0_proj_shortcut(relu_stage1_block3)
        conv_stage2_block0_branch2a = self.conv_stage2_block0_branch2a(relu_stage1_block3)
        bn_stage2_block0_proj_shortcut = self.bn_stage2_block0_proj_shortcut(conv_stage2_block0_proj_shortcut)
        bn_stage2_block0_branch2a = self.bn_stage2_block0_branch2a(conv_stage2_block0_branch2a)
        relu_stage2_block0_branch2a = self.relu_stage2_block0_branch2a(bn_stage2_block0_branch2a)
        conv_stage2_block0_branch2b_pad = self.conv_stage2_block0_branch2b_pad(relu_stage2_block0_branch2a, (1, 1, 1, 1))
        conv_stage2_block0_branch2b = self.conv_stage2_block0_branch2b(conv_stage2_block0_branch2b_pad)
        bn_stage2_block0_branch2b = self.bn_stage2_block0_branch2b(conv_stage2_block0_branch2b)
        relu_stage2_block0_branch2b = self.relu_stage2_block0_branch2b(bn_stage2_block0_branch2b)
        conv_stage2_block0_branch2c = self.conv_stage2_block0_branch2c(relu_stage2_block0_branch2b)
        bn_stage2_block0_branch2c = self.bn_stage2_block0_branch2c(conv_stage2_block0_branch2c)
        eltwise_stage2_block0 = self.eltwise_stage2_block0(bn_stage2_block0_proj_shortcut, bn_stage2_block0_branch2c)
        relu_stage2_block0 = self.relu_stage2_block0(eltwise_stage2_block0)
        conv_stage2_block1_branch2a = self.conv_stage2_block1_branch2a(relu_stage2_block0)
        bn_stage2_block1_branch2a = self.bn_stage2_block1_branch2a(conv_stage2_block1_branch2a)
        relu_stage2_block1_branch2a = self.relu_stage2_block1_branch2a(bn_stage2_block1_branch2a)
        conv_stage2_block1_branch2b_pad = self.conv_stage2_block1_branch2b_pad(relu_stage2_block1_branch2a, (1, 1, 1, 1))
        conv_stage2_block1_branch2b = self.conv_stage2_block1_branch2b(conv_stage2_block1_branch2b_pad)
        bn_stage2_block1_branch2b = self.bn_stage2_block1_branch2b(conv_stage2_block1_branch2b)
        relu_stage2_block1_branch2b = self.relu_stage2_block1_branch2b(bn_stage2_block1_branch2b)
        conv_stage2_block1_branch2c = self.conv_stage2_block1_branch2c(relu_stage2_block1_branch2b)
        bn_stage2_block1_branch2c = self.bn_stage2_block1_branch2c(conv_stage2_block1_branch2c)
        eltwise_stage2_block1 = self.eltwise_stage2_block1(relu_stage2_block0, bn_stage2_block1_branch2c)
        relu_stage2_block1 = self.relu_stage2_block1(eltwise_stage2_block1)
        conv_stage2_block2_branch2a = self.conv_stage2_block2_branch2a(relu_stage2_block1)
        bn_stage2_block2_branch2a = self.bn_stage2_block2_branch2a(conv_stage2_block2_branch2a)
        relu_stage2_block2_branch2a = self.relu_stage2_block2_branch2a(bn_stage2_block2_branch2a)
        conv_stage2_block2_branch2b_pad = self.conv_stage2_block2_branch2b_pad(relu_stage2_block2_branch2a, (1, 1, 1, 1))
        conv_stage2_block2_branch2b = self.conv_stage2_block2_branch2b(conv_stage2_block2_branch2b_pad)
        bn_stage2_block2_branch2b = self.bn_stage2_block2_branch2b(conv_stage2_block2_branch2b)
        relu_stage2_block2_branch2b = self.relu_stage2_block2_branch2b(bn_stage2_block2_branch2b)
        conv_stage2_block2_branch2c = self.conv_stage2_block2_branch2c(relu_stage2_block2_branch2b)
        bn_stage2_block2_branch2c = self.bn_stage2_block2_branch2c(conv_stage2_block2_branch2c)
        eltwise_stage2_block2 = self.eltwise_stage2_block2(relu_stage2_block1, bn_stage2_block2_branch2c)
        relu_stage2_block2 = self.relu_stage2_block2(eltwise_stage2_block2)
        conv_stage2_block3_branch2a = self.conv_stage2_block3_branch2a(relu_stage2_block2)
        bn_stage2_block3_branch2a = self.bn_stage2_block3_branch2a(conv_stage2_block3_branch2a)
        relu_stage2_block3_branch2a = self.relu_stage2_block3_branch2a(bn_stage2_block3_branch2a)
        conv_stage2_block3_branch2b_pad = self.conv_stage2_block3_branch2b_pad(relu_stage2_block3_branch2a, (1, 1, 1, 1))
        conv_stage2_block3_branch2b = self.conv_stage2_block3_branch2b(conv_stage2_block3_branch2b_pad)
        bn_stage2_block3_branch2b = self.bn_stage2_block3_branch2b(conv_stage2_block3_branch2b)
        relu_stage2_block3_branch2b = self.relu_stage2_block3_branch2b(bn_stage2_block3_branch2b)
        conv_stage2_block3_branch2c = self.conv_stage2_block3_branch2c(relu_stage2_block3_branch2b)
        bn_stage2_block3_branch2c = self.bn_stage2_block3_branch2c(conv_stage2_block3_branch2c)
        eltwise_stage2_block3 = self.eltwise_stage2_block3(relu_stage2_block2, bn_stage2_block3_branch2c)
        relu_stage2_block3 = self.relu_stage2_block3(eltwise_stage2_block3)
        conv_stage2_block4_branch2a = self.conv_stage2_block4_branch2a(relu_stage2_block3)
        bn_stage2_block4_branch2a = self.bn_stage2_block4_branch2a(conv_stage2_block4_branch2a)
        relu_stage2_block4_branch2a = self.relu_stage2_block4_branch2a(bn_stage2_block4_branch2a)
        conv_stage2_block4_branch2b_pad = self.conv_stage2_block4_branch2b_pad(relu_stage2_block4_branch2a, (1, 1, 1, 1))
        conv_stage2_block4_branch2b = self.conv_stage2_block4_branch2b(conv_stage2_block4_branch2b_pad)
        bn_stage2_block4_branch2b = self.bn_stage2_block4_branch2b(conv_stage2_block4_branch2b)
        relu_stage2_block4_branch2b = self.relu_stage2_block4_branch2b(bn_stage2_block4_branch2b)
        conv_stage2_block4_branch2c = self.conv_stage2_block4_branch2c(relu_stage2_block4_branch2b)
        bn_stage2_block4_branch2c = self.bn_stage2_block4_branch2c(conv_stage2_block4_branch2c)
        eltwise_stage2_block4 = self.eltwise_stage2_block4(relu_stage2_block3, bn_stage2_block4_branch2c)
        relu_stage2_block4 = self.relu_stage2_block4(eltwise_stage2_block4)
        conv_stage2_block5_branch2a = self.conv_stage2_block5_branch2a(relu_stage2_block4)
        bn_stage2_block5_branch2a = self.bn_stage2_block5_branch2a(conv_stage2_block5_branch2a)
        relu_stage2_block5_branch2a = self.relu_stage2_block5_branch2a(bn_stage2_block5_branch2a)
        conv_stage2_block5_branch2b_pad = self.conv_stage2_block5_branch2b_pad(relu_stage2_block5_branch2a, (1, 1, 1, 1))
        conv_stage2_block5_branch2b = self.conv_stage2_block5_branch2b(conv_stage2_block5_branch2b_pad)
        bn_stage2_block5_branch2b = self.bn_stage2_block5_branch2b(conv_stage2_block5_branch2b)
        relu_stage2_block5_branch2b = self.relu_stage2_block5_branch2b(bn_stage2_block5_branch2b)
        conv_stage2_block5_branch2c = self.conv_stage2_block5_branch2c(relu_stage2_block5_branch2b)
        bn_stage2_block5_branch2c = self.bn_stage2_block5_branch2c(conv_stage2_block5_branch2c)
        eltwise_stage2_block5 = self.eltwise_stage2_block5(relu_stage2_block4, bn_stage2_block5_branch2c)
        relu_stage2_block5 = self.relu_stage2_block5(eltwise_stage2_block5)
        conv_stage3_block0_proj_shortcut = self.conv_stage3_block0_proj_shortcut(relu_stage2_block5)
        conv_stage3_block0_branch2a = self.conv_stage3_block0_branch2a(relu_stage2_block5)
        bn_stage3_block0_proj_shortcut = self.bn_stage3_block0_proj_shortcut(conv_stage3_block0_proj_shortcut)
        bn_stage3_block0_branch2a = self.bn_stage3_block0_branch2a(conv_stage3_block0_branch2a)
        relu_stage3_block0_branch2a = self.relu_stage3_block0_branch2a(bn_stage3_block0_branch2a)
        conv_stage3_block0_branch2b_pad = self.conv_stage3_block0_branch2b_pad(relu_stage3_block0_branch2a, (1, 1, 1, 1))
        conv_stage3_block0_branch2b = self.conv_stage3_block0_branch2b(conv_stage3_block0_branch2b_pad)
        bn_stage3_block0_branch2b = self.bn_stage3_block0_branch2b(conv_stage3_block0_branch2b)
        relu_stage3_block0_branch2b = self.relu_stage3_block0_branch2b(bn_stage3_block0_branch2b)
        conv_stage3_block0_branch2c = self.conv_stage3_block0_branch2c(relu_stage3_block0_branch2b)
        bn_stage3_block0_branch2c = self.bn_stage3_block0_branch2c(conv_stage3_block0_branch2c)
        eltwise_stage3_block0 = self.eltwise_stage3_block0(bn_stage3_block0_proj_shortcut, bn_stage3_block0_branch2c)
        relu_stage3_block0 = self.relu_stage3_block0(eltwise_stage3_block0)
        conv_stage3_block1_branch2a = self.conv_stage3_block1_branch2a(relu_stage3_block0)
        bn_stage3_block1_branch2a = self.bn_stage3_block1_branch2a(conv_stage3_block1_branch2a)
        relu_stage3_block1_branch2a = self.relu_stage3_block1_branch2a(bn_stage3_block1_branch2a)
        conv_stage3_block1_branch2b_pad = self.conv_stage3_block1_branch2b_pad(relu_stage3_block1_branch2a, (1, 1, 1, 1))
        conv_stage3_block1_branch2b = self.conv_stage3_block1_branch2b(conv_stage3_block1_branch2b_pad)
        bn_stage3_block1_branch2b = self.bn_stage3_block1_branch2b(conv_stage3_block1_branch2b)
        relu_stage3_block1_branch2b = self.relu_stage3_block1_branch2b(bn_stage3_block1_branch2b)
        conv_stage3_block1_branch2c = self.conv_stage3_block1_branch2c(relu_stage3_block1_branch2b)
        bn_stage3_block1_branch2c = self.bn_stage3_block1_branch2c(conv_stage3_block1_branch2c)
        eltwise_stage3_block1 = self.eltwise_stage3_block1(relu_stage3_block0, bn_stage3_block1_branch2c)
        relu_stage3_block1 = self.relu_stage3_block1(eltwise_stage3_block1)
        conv_stage3_block2_branch2a = self.conv_stage3_block2_branch2a(relu_stage3_block1)
        bn_stage3_block2_branch2a = self.bn_stage3_block2_branch2a(conv_stage3_block2_branch2a)
        relu_stage3_block2_branch2a = self.relu_stage3_block2_branch2a(bn_stage3_block2_branch2a)
        conv_stage3_block2_branch2b_pad = self.conv_stage3_block2_branch2b_pad(relu_stage3_block2_branch2a, (1, 1, 1, 1))
        conv_stage3_block2_branch2b = self.conv_stage3_block2_branch2b(conv_stage3_block2_branch2b_pad)
        bn_stage3_block2_branch2b = self.bn_stage3_block2_branch2b(conv_stage3_block2_branch2b)
        relu_stage3_block2_branch2b = self.relu_stage3_block2_branch2b(bn_stage3_block2_branch2b)
        conv_stage3_block2_branch2c = self.conv_stage3_block2_branch2c(relu_stage3_block2_branch2b)
        bn_stage3_block2_branch2c = self.bn_stage3_block2_branch2c(conv_stage3_block2_branch2c)
        eltwise_stage3_block2 = self.eltwise_stage3_block2(relu_stage3_block1, bn_stage3_block2_branch2c)
        relu_stage3_block2 = self.relu_stage3_block2(eltwise_stage3_block2)
        avgpool_2d = nn.AdaptiveAvgPool2d((7, 7))
        relu_stage3_block2 = avgpool_2d(relu_stage3_block2)
        pool = F.avg_pool2d(relu_stage3_block2, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        fc_nsfw_0 = pool.view(pool.size(0), -1)
        fc_nsfw_1 = self.fc_nsfw_1(fc_nsfw_0)
        prob = self.prob(fc_nsfw_1, dim=1)
        return prob


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AVGPoolLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (AdditionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelMod,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Classify,
     lambda: ([], {'labels': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DropoutLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Jitter,
     lambda: ([], {'jitter_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (L2Regularizer,
     lambda: ([], {'strength': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalResponseNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaxPool2dLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModelPlus,
     lambda: ([], {'input_net': _mock_layer(), 'net': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PadLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RandomTransform,
     lambda: ([], {'t_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReluLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftMaxLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TVLoss,
     lambda: ([], {'strength': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ProGamerGov_neural_dream(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

