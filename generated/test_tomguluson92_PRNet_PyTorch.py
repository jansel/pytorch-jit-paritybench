import sys
_module = sys.modules[__name__]
del sys
api = _module
config = _module
face3d = _module
mesh = _module
setup = _module
io = _module
light = _module
render = _module
transform = _module
vis = _module
mesh_numpy = _module
morphable_model = _module
fit = _module
load = _module
morphabel_model = _module
inference = _module
model = _module
resfcn256 = _module
WLP300dataset = _module
prnet_loss = _module
train = _module
cv_plot = _module
estimate_pose = _module
generate_posmap_300WLP = _module
losses = _module
rotate_vertices = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torchvision.models import *


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torchvision.transforms.functional as F


import random


import numbers


import torch.optim


from torch.utils.tensorboard import SummaryWriter


from torchvision import utils


from torchvision import models


from torch.autograd import Variable


from torch.nn import Conv2d


from math import exp


import math


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.shortcut_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, stride=1, padding=0)
        self.normalizer_fn = norm_layer(planes)
        self.activation_fn = nn.ReLU(inplace=True)
        self.stride = stride
        self.out_planes = planes

    def forward(self, x):
        shortcut = x
        _, _, _, x_planes = x.size()
        if self.stride != 1 or x_planes != self.out_planes:
            shortcut = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += shortcut
        x = self.normalizer_fn(x)
        x = self.activation_fn(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding='same'):
    """3x3 convolution with padding"""
    if padding == 'same':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=dilation)


class ResFCN256(nn.Module):

    def __init__(self, resolution_input=256, resolution_output=256, channel=3, size=16):
        super().__init__()
        self.input_resolution = resolution_input
        self.output_resolution = resolution_output
        self.channel = channel
        self.size = size
        self.block0 = conv3x3(in_planes=3, out_planes=self.size, padding='same')
        self.block1 = ResBlock(inplanes=self.size, planes=self.size * 2, stride=2)
        self.block2 = ResBlock(inplanes=self.size * 2, planes=self.size * 2, stride=1)
        self.block3 = ResBlock(inplanes=self.size * 2, planes=self.size * 4, stride=2)
        self.block4 = ResBlock(inplanes=self.size * 4, planes=self.size * 4, stride=1)
        self.block5 = ResBlock(inplanes=self.size * 4, planes=self.size * 8, stride=2)
        self.block6 = ResBlock(inplanes=self.size * 8, planes=self.size * 8, stride=1)
        self.block7 = ResBlock(inplanes=self.size * 8, planes=self.size * 16, stride=2)
        self.block8 = ResBlock(inplanes=self.size * 16, planes=self.size * 16, stride=1)
        self.block9 = ResBlock(inplanes=self.size * 16, planes=self.size * 32, stride=2)
        self.block10 = ResBlock(inplanes=self.size * 32, planes=self.size * 32, stride=1)
        self.upsample0 = nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
        self.upsample4 = nn.ConvTranspose2d(self.size * 16, self.size * 8, kernel_size=4, stride=2, padding=1)
        self.upsample5 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1, padding=1)
        self.upsample6 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1, padding=1)
        self.upsample7 = nn.ConvTranspose2d(self.size * 8, self.size * 4, kernel_size=4, stride=2, padding=1)
        self.upsample8 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1, padding=1)
        self.upsample9 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1, padding=1)
        self.upsample10 = nn.ConvTranspose2d(self.size * 4, self.size * 2, kernel_size=4, stride=2, padding=1)
        self.upsample11 = nn.ConvTranspose2d(self.size * 2, self.size * 2, kernel_size=3, stride=1, padding=1)
        self.upsample12 = nn.ConvTranspose2d(self.size * 2, self.size, kernel_size=4, stride=2, padding=1)
        self.upsample13 = nn.ConvTranspose2d(self.size, self.size, kernel_size=3, stride=1, padding=1)
        self.upsample14 = nn.ConvTranspose2d(self.size, self.channel, kernel_size=3, stride=1, padding=1)
        self.upsample15 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
        self.upsample16 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.block0(x)
        se = self.block1(se)
        se = self.block2(se)
        se = self.block3(se)
        se = self.block4(se)
        se = self.block5(se)
        se = self.block6(se)
        se = self.block7(se)
        se = self.block8(se)
        se = self.block9(se)
        se = self.block10(se)
        pd = self.upsample0(se)
        pd = self.upsample1(pd)
        pd = self.upsample2(pd)
        pd = self.upsample3(pd)
        pd = self.upsample4(pd)
        pd = self.upsample5(pd)
        pd = self.upsample6(pd)
        pd = self.upsample7(pd)
        pd = self.upsample8(pd)
        pd = self.upsample9(pd)
        pd = self.upsample10(pd)
        pd = self.upsample11(pd)
        pd = self.upsample12(pd)
        pd = self.upsample13(pd)
        pd = self.upsample14(pd)
        pd = self.upsample15(pd)
        pos = self.upsample16(pd)
        pos = self.sigmoid(pos)
        return pos


def preprocess(mask):
    """
    :param mask: grayscale of mask.
    :return:
    """
    tmp = {}
    mask[mask > 0] = mask[mask > 0] / 16
    mask[mask == 15] = 16
    mask[mask == 7] = 8
    return mask


class WeightMaskLoss(nn.Module):
    """
        L2_Loss * Weight Mask
    """

    def __init__(self, mask_path):
        super(WeightMaskLoss, self).__init__()
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float()
        else:
            raise FileNotFoundError('Mask File Not Found! Please Check your Settings!')

    def forward(self, pred, gt):
        result = torch.mean(torch.pow(pred - gt, 2), dim=1)
        result = torch.mul(result, self.mask)
        result = torch.sum(result)
        result = result / self.mask.size(1) ** 2
        return result


def _fspecial_gauss(window_size, sigma=1.5):
    coords = np.arange(0, window_size, dtype=np.float32)
    coords -= (window_size - 1) / 2.0
    g = coords ** 2
    g *= -0.5 / sigma ** 2
    g = np.reshape(g, (1, -1)) + np.reshape(g, (-1, 1))
    g = torch.from_numpy(np.reshape(g, (1, -1)))
    g = torch.softmax(g, dim=1)
    g = g / g.sum()
    return g


def butterworth(window_size, sigma=1.5, n=2):
    nn = 2 * n
    bw = torch.Tensor([(1 / (1 + ((x - window_size // 2) / sigma) ** nn)) for x in range(window_size)])
    return bw / bw.sum()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([(init_dim * np.arange(n_tile) + i) for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def create_window(window_size, channel=3, sigma=1.5, gauss='original', n=2):
    if gauss == 'original':
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    elif gauss == 'butterworth':
        _1D_window = butterworth(window_size, sigma, n).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    else:
        g = _fspecial_gauss(window_size, sigma)
        g = torch.reshape(g, (1, 1, window_size, window_size))
        g = tile(g, 0, 3)
        return g


def _ssim(img1, img2, window_size=11, window=None, val_range=2, size_average=True):
    padd = 0
    batch, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_square = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_square = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12_square = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12_square + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_square + sigma2_square + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class ORIGINAL_SSIM(torch.nn.Module):

    def __init__(self, window_size=11, val_range=2, size_average=True):
        super(ORIGINAL_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, self.window_size, window, self.val_range, self.size_average)


def dfl_ssim(img1, img2, mask, window_size=11, val_range=1, gauss='original'):
    padd = 0
    batch, channel, height, width = img1.size()
    img1, img2 = torch.mul(img1, mask), torch.mul(img2, mask)
    real_size = min(window_size, height, width)
    window = create_window(real_size, gauss=gauss)
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    num0 = mu1 * mu2 * 2.0
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    den0 = mu1_sq + mu2_sq
    luminance = (num0 + c1) / (den0 + c1)
    num1 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) * 2.0
    den1 = F.conv2d(img1 * img1 + img2 * img2, window, padding=padd, groups=channel)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim_val = torch.mean(luminance * cs, dim=(-3, -2))
    return torch.mean((1.0 - ssim_val) / 2.0)


class SSIM(torch.nn.Module):

    def __init__(self, mask_path, window_size=11, alpha=0.8, gauss='original'):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.window = None
        self.channel = None
        self.gauss = gauss
        self.alpha = alpha
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float()
        else:
            raise FileNotFoundError('Mask File Not Found! Please Check your Settings!')

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        self.channel = channel
        return 10 * dfl_ssim(img1, img2, mask=self.mask, window_size=self.window_size, gauss=self.gauss)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResFCN256,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_tomguluson92_PRNet_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

