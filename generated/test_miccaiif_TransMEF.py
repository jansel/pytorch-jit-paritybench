import sys
_module = sys.modules[__name__]
del sys
Network_TransMEF = _module
dataloader_TransMEF = _module
fusion_arbitary_size_TransMEF_gray = _module
fusion_gray_TransMEF = _module
log = _module
resize_all = _module
ssim = _module
train_TransMEF = _module
utils = _module

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


import torch.nn as nn


from torch import einsum


import torch.utils.data as Data


import torchvision.transforms as transforms


import numpy as np


import copy


import random


from collections import OrderedDict


import string


import time


import sklearn.metrics as metrics


import torch.utils.data as data


import torch.nn.functional as F


from torch.autograd import Variable


from math import exp


from torch.utils.data.sampler import SubsetRandomSampler


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.data import DataLoader


import torch.optim as optim


import torchvision


from torchvision import transforms


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """1*1 conv before the output"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def save_grad(grads, name):

    def hook(grad):
        grads[name] = grad
    return hook


class Encoder(nn.Module):
    """features extraction"""

    def __init__(self):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(16, 32)
        self.layer2 = DoubleConv(32, 48)

    def forward(self, x, grads=None, name=None):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if grads is not None:
            x.register_hook(save_grad(grads, name + '_x'))
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.dim = dim
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.convd1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=16, c=1)(x)
        return x


class Encoder_Trans(nn.Module):
    """features extraction"""

    def __init__(self):
        super(Encoder_Trans, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(17, 32)
        self.layer2 = DoubleConv(32, 48)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)

    def forward(self, x, grads=None, name=None):
        x_e = self.inc(x)
        x_t = self.transformer(x)
        x = torch.cat((x_e, x_t), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        if grads is not None:
            x.register_hook(save_grad(grads, name + '_x'))
        return x


class Decoder(nn.Module):
    """reconstruction"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DoubleConv(48, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.outc(x)
        return output


class Decoder_Trans(nn.Module):
    """reconstruction"""

    def __init__(self):
        super(Decoder_Trans, self).__init__()
        self.layer3 = DoubleConv(49, 48)
        self.layer4 = DoubleConv(48, 48)
        self.layer1 = DoubleConv(48, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer4(self.layer3(x))
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.outc(x)
        return output


class SimNet(nn.Module):
    """easy network for self-reconstruction task"""

    def __init__(self):
        super(SimNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TransNet(nn.Module):
    """U-based network for self-reconstruction task"""

    def __init__(self):
        super(TransNet, self).__init__()
        self.encoder = Encoder_Trans()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Strategy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y1, y2):
        return (y1 + y2) / 2


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow(r[:, :, 1:, :] - r[:, :, :h - 1, :], 2).mean()
        tv2 = torch.pow(r[:, :, :, 1:] - r[:, :, :, :w - 1], 2).mean()
        return tv1 + tv2


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
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
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 48, 64, 64])], {}),
     True),
    (Decoder_Trans,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 49, 64, 64])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Strategy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TV_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_miccaiif_TransMEF(_paritybench_base):
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

