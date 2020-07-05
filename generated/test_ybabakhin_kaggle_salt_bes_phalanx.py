import sys
_module = sys.modules[__name__]
del sys
augmentations = _module
callbacks = _module
snapshot = _module
datasets = _module
generate_folds = _module
generators = _module
ensemble = _module
losses = _module
models = _module
models_zoo = _module
params = _module
predict_test = _module
segmentation_models = _module
__version__ = _module
backbones = _module
classification_models = _module
resnet = _module
blocks = _module
builder = _module
preprocessing = _module
resnext = _module
utils = _module
weights = _module
tests = _module
test_imagenet = _module
inception_resnet_v2 = _module
inception_v3 = _module
common = _module
functions = _module
layers = _module
fpn = _module
model = _module
linknet = _module
pspnet = _module
unet = _module
train = _module
lovasz_losses = _module
make_pseudo = _module
precisioncv = _module
salt_dataset = _module
submit34 = _module
train_cv = _module
train_pseudo = _module
transform = _module
unet_model = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data.sampler import RandomSampler


import torchvision


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class FPAv2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))
        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))

    def forward(self, x):
        x_glob = self.glob(x)
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)
        d2 = self.down2_1(x)
        d3 = self.down3_1(d2)
        d2 = self.down2_2(d2)
        d3 = self.down3_2(d3)
        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = d2 + d3
        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = x * d2
        x = x + x_glob
        return x


class SpatialAttention2d(nn.Module):

    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):

    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class Decoderv2(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class SCse(nn.Module):

    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class Res34Unetv4(nn.Module):

    def __init__(self):
        super(Res34Unetv4, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))
        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))
        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)
        self.logit = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ELU(True), nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.conv1(x)
        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        f = self.center(e5)
        d5 = self.decode5(f, e5)
        d4 = self.decode4(d5, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2)
        f = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True), F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True), F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True), F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)
        logit = self.logit(f)
        return logit


class Res34Unetv3(nn.Module):

    def __init__(self):
        super(Res34Unetv3, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))
        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))
        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)
        self.dropout2d = nn.Dropout2d(0.4)
        self.dropout = nn.Dropout(0.4)
        self.fuse_pixel = conv3x3(320, 64)
        self.logit_pixel = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.fuse_image = nn.Sequential(nn.Linear(512, 64), nn.ELU(True))
        self.logit_image = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.logit = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False), nn.ELU(True), nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x = self.conv1(x)
        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        e = self.dropout(e)
        f = self.center(e5)
        d5 = self.decode5(f, e5)
        d4 = self.decode4(d5, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2)
        f = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True), F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True), F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True), F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)
        f = self.dropout2d(f)
        fuse_pixel = self.fuse_pixel(f)
        logit_pixel = self.logit_pixel(fuse_pixel)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image)
        fuse = torch.cat([fuse_pixel, F.upsample(fuse_image.view(batch_size, -1, 1, 1), scale_factor=256, mode='bilinear', align_corners=True)], 1)
        logit = self.logit(fuse)
        return logit, logit_pixel, logit_image.view(-1)


class Res34Unetv5(nn.Module):

    def __init__(self):
        super(Res34Unetv5, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), self.resnet.bn1, self.resnet.relu)
        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))
        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))
        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.logit = nn.Sequential(nn.Conv2d(256, 32, kernel_size=3, padding=1), nn.ELU(True), nn.Conv2d(32, 1, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.conv1(x)
        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        f = self.center(e5)
        d5 = self.decode5(f, e5)
        d4 = self.decode4(d5, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        f = torch.cat((d2, F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True), F.upsample(d4, scale_factor=4, mode='bilinear', align_corners=True), F.upsample(d5, scale_factor=8, mode='bilinear', align_corners=True)), 1)
        f = F.dropout2d(f, p=0.4)
        logit = self.logit(f)
        return logit


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'out_channels': 64}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoderv2,
     lambda: ([], {'up_in': 4, 'x_in': 4, 'n_out': 64}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {}),
     True),
    (FPAv2,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 16, 16])], {}),
     False),
    (GAB,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SCse,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialAttention2d,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ybabakhin_kaggle_salt_bes_phalanx(_paritybench_base):
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

