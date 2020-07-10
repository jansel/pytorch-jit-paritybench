import sys
_module = sys.modules[__name__]
del sys
data = _module
model = _module
test = _module
train = _module
train_text_embedding = _module

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
xrange = range
wraps = functools.wraps


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision.models as models


from torchvision.utils import save_image


import torch.optim.lr_scheduler as lr_scheduler


class VisualSemanticEmbedding(nn.Module):

    def __init__(self, embed_ndim):
        super(VisualSemanticEmbedding, self).__init__()
        self.embed_ndim = embed_ndim
        self.img_encoder = models.vgg16(pretrained=True)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.feat_extractor = nn.Sequential(*(self.img_encoder.classifier[i] for i in range(6)))
        self.W = nn.Linear(4096, embed_ndim, False)
        self.txt_encoder = nn.GRU(embed_ndim, embed_ndim, 1)

    def forward(self, img, txt):
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.feat_extractor(img_feat)
        img_feat = self.W(img_feat)
        h0 = torch.zeros(1, img.size(0), self.embed_ndim)
        h0 = Variable(h0 if txt.data.is_cuda else h0)
        _, txt_feat = self.txt_encoder(txt, h0)
        txt_feat = txt_feat.squeeze()
        return img_feat, txt_feat


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512))

    def forward(self, x):
        return F.relu(x + self.encoder(x))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, use_vgg=True):
        super(Generator, self).__init__()
        if use_vgg:
            self.encoder = models.vgg16_bn(pretrained=True)
            self.encoder = nn.Sequential(*(self.encoder.features[i] for i in range(23) + range(24, 33)))
            self.encoder[24].dilation = 2, 2
            self.encoder[24].padding = 2, 2
            self.encoder[27].dilation = 2, 2
            self.encoder[27].padding = 2, 2
            self.encoder[30].dilation = 2, 2
            self.encoder[30].padding = 2, 2
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder = nn.Sequential(nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 4, 2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 512, 4, 2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.residual_blocks = nn.Sequential(nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock())
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 3, 3, padding=1), nn.Tanh())
        self.mu = nn.Sequential(nn.Linear(300, 128, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.log_sigma = nn.Sequential(nn.Linear(300, 128, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, img, txt_feat, z=None):
        img_feat = self.encoder(img)
        z_mean = self.mu(txt_feat)
        z_log_stddev = self.log_sigma(txt_feat)
        z = torch.randn(txt_feat.size(0), 128)
        if next(self.parameters()).is_cuda:
            z = z
        txt_feat = z_mean + z_log_stddev.exp() * Variable(z)
        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        fusion = torch.cat((img_feat, txt_feat), dim=1)
        fusion = self.residual_blocks(fusion)
        output = self.decoder(fusion)
        return output, (z_mean, z_log_stddev)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 4, 2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, 4, 2, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, 4, 2, padding=1, bias=False), nn.BatchNorm2d(512))
        self.residual_branch = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512))
        self.classifier = nn.Sequential(nn.Conv2d(512 + 128, 512, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 1, 4))
        self.compression = nn.Sequential(nn.Linear(300, 128), nn.LeakyReLU(0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, img, txt_feat):
        img_feat = self.encoder(img)
        img_feat = F.leaky_relu(img_feat + self.residual_branch(img_feat), 0.2)
        txt_feat = self.compression(txt_feat)
        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        fusion = torch.cat((img_feat, txt_feat), dim=1)
        output = self.classifier(fusion)
        return output.squeeze()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 300])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (VisualSemanticEmbedding,
     lambda: ([], {'embed_ndim': 4}),
     lambda: ([torch.rand([4, 3, 243, 243]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_woozzu_dong_iccv_2017(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

