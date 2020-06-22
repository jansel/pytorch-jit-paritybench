import sys
_module = sys.modules[__name__]
del sys
app_utils = _module
display = _module
producer = _module
train = _module
crops_train = _module
datasets = _module
labels = _module
losses = _module
metrics = _module
models = _module
optim = _module
summary_utils = _module
train_utils = _module
bbox_transforms = _module
color_tables = _module
inferno = _module
viridis = _module
colormaps = _module
crops = _module
exceptions = _module
image_utils = _module
load_baseline = _module
profiling = _module
tensor_conv = _module
visualization = _module
vis_app = _module

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


from math import floor


import numpy as np


from collections import namedtuple


import torch


import torch.nn.functional as F


from torch import sigmoid


import logging


from torch.utils.data import DataLoader


import math


import torch.nn as nn


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import zeros_


from torch.nn.init import normal_


import torch.optim as optim


from scipy import io


class BaselineEmbeddingNet(nn.Module):
    """ Definition of the embedding network used in the baseline experiment of
    Bertinetto et al in https://arxiv.org/pdf/1704.06036.pdf.
    It basically corresponds to the convolutional stage of AlexNet, with some
    of its hyperparameters changed.
    """

    def __init__(self):
        super(BaselineEmbeddingNet, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11,
            stride=2, bias=True), nn.BatchNorm2d(96), nn.ReLU(), nn.
            MaxPool2d(3, stride=2), nn.Conv2d(96, 256, kernel_size=5,
            stride=1, groups=2, bias=True), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(3, stride=1), nn.Conv2d(256, 384, kernel_size=3,
            stride=1, groups=1, bias=True), nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, bias=
            True), nn.BatchNorm2d(384), nn.ReLU(), nn.Conv2d(384, 32,
            kernel_size=3, stride=1, groups=2, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class VGG11EmbeddingNet_5c(nn.Module):
    """ Embedding branch based on Pytorch's VGG11 with Batchnorm (https://pytor
    ch.org/docs/stable/torchvision/models.html). This is version 5c, meaning
    that it has 5 convolutional layers, it follows the original model up until
    the 13th layer (The ReLU after the 4th convolution), in order to keep the
    total stride equal to 4. It adds the 5th convolutional layer which acts as
    a bottleck a feature bottleneck reducing the features from 256 to 32 and
    must always be trained. The layers 0 to 13 can be loaded from
    torchvision.models.vgg11_bn(pretrained=True)
    """

    def __init__(self):
        super(VGG11EmbeddingNet_5c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            stride=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(), nn.
            MaxPool2d(2, stride=2), nn.Conv2d(64, 128, kernel_size=3,
            stride=1, bias=True), nn.BatchNorm2d(128), nn.ReLU(), nn.
            MaxPool2d(2, stride=2), nn.Conv2d(128, 256, kernel_size=3,
            stride=1, bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d
            (256, 256, kernel_size=3, stride=1, bias=True), nn.BatchNorm2d(
            256), nn.ReLU(), nn.Conv2d(256, 32, kernel_size=3, stride=1,
            bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class VGG16EmbeddingNet_8c(nn.Module):
    """ Embedding branch based on Pytorch's VGG16 with Batchnorm (https://pytor
    ch.org/docs/stable/torchvision/models.html). This is version 8c, meaning
    that it has 8 convolutional layers, it follows the original model up until
    the 22th layer (The ReLU after the 7th convolution), in order to keep the
    total stride equal to 4. It adds the 8th convolutional layer which acts as
    a bottleck a feature bottleneck reducing the features from 256 to 32 and
    must always be trained. The layers 0 to 22 can be loaded from
    torchvision.models.vgg16_bn(pretrained=True)
    """

    def __init__(self):
        super(VGG16EmbeddingNet_8c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            stride=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(
            64, 64, kernel_size=3, stride=1, bias=True), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d(64, 128,
            kernel_size=3, stride=1, bias=True), nn.BatchNorm2d(128), nn.
            ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.
            Conv2d(128, 256, kernel_size=3, stride=1, bias=True), nn.
            BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3,
            stride=1, bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d
            (256, 256, kernel_size=3, stride=1, bias=True), nn.BatchNorm2d(
            256), nn.ReLU(), nn.Conv2d(256, 32, kernel_size=3, stride=1,
            bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    """ The basic siamese network joining network, that takes the outputs of
    two embedding branches and joins them applying a correlation operation.
    Should always be used with tensors of the form [B x C x H x W], i.e.
    you must always include the batch dimension.
    """

    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4
        ):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.upscale = upscale
        self.corr_map_size = corr_map_size
        self.stride = stride
        self.upsc_size = (self.corr_map_size - 1) * self.stride + 1
        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)

    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref,
            groups=b)
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode=
                'bilinear', align_corners=False)
        return match_map


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rafellerc_Pytorch_SiamFC(_paritybench_base):
    pass
    def test_000(self):
        self._check(BaselineEmbeddingNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(SiameseNet(*[], **{'embedding_net': ReLU()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(VGG11EmbeddingNet_5c(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_003(self):
        self._check(VGG16EmbeddingNet_8c(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

