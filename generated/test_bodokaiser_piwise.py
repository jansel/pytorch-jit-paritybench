import sys
_module = sys.modules[__name__]
del sys
main = _module
criterion = _module
dataset = _module
network = _module
transform = _module
visualize = _module

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


import numpy as np


import torch


from torch.optim import SGD


from torch.optim import Adam


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torchvision.transforms import Compose


from torchvision.transforms import CenterCrop


from torchvision.transforms import Normalize


from torchvision.transforms import ToTensor


from torchvision.transforms import ToPILImage


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


import torch.nn.init as init


from torch.utils import model_zoo


from torchvision import models


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


class FCN8(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False
        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout())
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)
        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)
        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        score = F.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3
        return F.upsample_bilinear(score, x.size()[2:])


class FCN16(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout())
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)
        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        return F.upsample_bilinear(score, x.size()[2:])


class FCN32(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout())
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)
        return F.upsample_bilinear(score, x.size()[2:])


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, features, 3), nn.ReLU(inplace=True), nn.Conv2d(features, features, 3), nn.ReLU(inplace=True), nn.ConvTranspose2d(features, out_channels, 2, stride=2), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3), nn.ReLU(inplace=True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 3), nn.ReLU(inplace=True), nn.Dropout(), nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ReLU(inplace=True))
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))
        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = [nn.Conv2d(in_channels, in_channels // 2, 3, padding=1), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True)] * num_layers
        layers += [nn.Conv2d(in_channels // 2, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features
        self.dec1 = features[0:4]
        self.dec2 = features[5:9]
        self.dec3 = features[10:16]
        self.dec4 = features[17:23]
        self.dec5 = features[24:-1]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False
        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(512, 256, 1)
        self.enc3 = SegNetEnc(256, 128, 1)
        self.enc2 = SegNetEnc(128, 64, 0)
        self.final = nn.Sequential(*[nn.Conv2d(64, classes, 3, padding=1), nn.BatchNorm2d(classes), nn.ReLU(inplace=True)])

    def forward(self, x):
        x1 = self.dec1(x)
        d1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.dec2(d1)
        d2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.dec3(d2)
        d3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.dec4(d3)
        d4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.dec5(d4)
        d5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        def upsample(d):
            e5 = self.enc5(F.max_unpool2d(d, m5, kernel_size=2, stride=2, output_size=x5.size()))
            e4 = self.enc4(F.max_unpool2d(e5, m4, kernel_size=2, stride=2, output_size=x4.size()))
            e3 = self.enc3(F.max_unpool2d(e4, m3, kernel_size=2, stride=2, output_size=x3.size()))
            e2 = self.enc2(F.max_unpool2d(e3, m2, kernel_size=2, stride=2, output_size=x2.size()))
            e1 = F.max_unpool2d(e2, m1, kernel_size=2, stride=2, output_size=x1.size())
            return e1
        e = upsample(d5)
        return self.final(e)


class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=60):
        super().__init__()
        self.features = nn.Sequential(nn.AvgPool2d(downsize, stride=downsize), nn.Conv2d(in_features, out_features, 1, bias=False), nn.BatchNorm2d(out_features, momentum=0.95), nn.ReLU(inplace=True), nn.UpsamplingBilinear2d(upsize))

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        """
        resnet = models.resnet101(pretrained=True)
        self.conv1 = resnet.conv1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False
        self.layer5a = PSPDec(2048, 512, 60)
        self.layer5b = PSPDec(2048, 512, 30)
        self.layer5c = PSPDec(2048, 512, 20)
        self.layer5d = PSPDec(2048, 512, 10)
        self.final = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512, momentum=0.95), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(512, num_classes, 1))

    def forward(self, x):
        None
        x = self.conv1(x)
        None
        x = self.layer1(x)
        None
        x = self.layer2(x)
        None
        x = self.layer3(x)
        None
        x = self.layer4(x)
        None
        x = self.final(torch.cat([x, self.layer5a(x), self.layer5b(x), self.layer5c(x), self.layer5d(x)], 1))
        None
        return F.upsample_bilinear(final, x.size()[2:])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FCN16,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FCN32,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (FCN8,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PSPDec,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'downsize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SegNet,
     lambda: ([], {'classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SegNetEnc,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (UNetDec,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (UNetEnc,
     lambda: ([], {'in_channels': 4, 'features': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_bodokaiser_piwise(_paritybench_base):
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

