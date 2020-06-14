import sys
_module = sys.modules[__name__]
del sys
Dataloader = _module
demo_segmentation = _module
loss = _module
ACNN = _module
BaseModels = _module
MobileNetV2 = _module
Xception = _module
models = _module
common = _module
image_inpainting = _module
partial_convolution = _module
text_segmentation = _module
cls = _module

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


import random


import re


from itertools import chain


import numpy as np


import torch


from torch import nn


from torch.nn.functional import pad


from torch.utils.data import Dataset


import time


from torch.nn import functional as F


from torch.nn.parameter import Parameter


import math


from torch.nn.functional import affine_grid


from torch.nn.functional import grid_sample


from torch.utils.checkpoint import checkpoint


from torch.nn.functional import avg_pool2d


class MultiClassFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryFocalLoss(nn.BCEWithLogitsLoss):

    def __init__(self, gamma=0, background_weights=1, words_weights=2):
        super(BinaryFocalLoss, self).__init__(size_average=True, reduce=False)
        self.gamma = gamma
        self.background_weights = background_weights
        self.words_weights = words_weights

    def forward(self, input, target):
        input = self.flatten_images(input)
        target = self.flatten_images(target)
        weights = torch.where(target > 0, torch.ones_like(target) * self.
            words_weights, torch.ones_like(target) * self.background_weights)
        pt = F.logsigmoid(-input * (target * 2 - 1))
        loss = F.binary_cross_entropy_with_logits(input, target, weight=
            weights, size_average=True, reduce=False)
        loss = (pt * self.gamma).exp() * loss
        return loss.mean()

    @staticmethod
    def flatten_images(x):
        assert x.dim() == 4 and x.size(1) == 1
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2)
        x = x.contiguous().view(-1, x.size(2))
        return x


class SoftBootstrapCrossEntropy(nn.BCELoss):
    """
    TRAINING DEEP NEURAL NETWORKS ON NOISY LABELS WITH BOOTSTRAPPING (https://arxiv.org/pdf/1412.6596.pdf)
    # Tensorflow: https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/object_detection/core/losses.py#L275-L336
    ++++   Use with caution ++++
    with this loss, the model may learn to detect words that are not labeled.
    but  not all words are necessary whited out
    """

    def __init__(self, beta=0.95, background_weight=1, words_weight=2,
        size_average=True, reduce=True):
        super(SoftBootstrapCrossEntropy, self).__init__(size_average=
            size_average, reduce=reduce)
        self.beta = beta
        self.background_weight = background_weight
        self.words_weight = words_weight
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        input = self.flatten_images(input)
        target = self.flatten_images(target)
        weights = torch.where(target > 0, torch.ones_like(target) * self.
            words_weight, torch.ones_like(target) * self.background_weight)
        bootstrap_target = self.beta * target + (1 - self.beta) * (F.
            sigmoid(input) > 0.5).float()
        return F.binary_cross_entropy_with_logits(input, bootstrap_target,
            weight=weights, size_average=self.size_average, reduce=self.reduce)

    @staticmethod
    def flatten_images(x):
        assert x.dim() == 4 and x.size(1) == 1
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2)
        x = x.contiguous().view(-1, x.size(2))
        return x


use_cuda = torch.cuda.is_available()


FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class BCERegionLoss(nn.Module):

    def __init__(self):
        super(BCERegionLoss, self).__init__()
        self.anchor_box = FloatTensor([(0.4, 0.4), (0.4, -0.4), (-0.4, -0.4
            ), (-0.4, 0.4)]).unsqueeze(-1)
        self.scale_alpha = FloatTensor([1])
        self.positive_beta = FloatTensor([0.2])
        self.bce = nn.BCEWithLogitsLoss()

    def scale_loss(self, scale):
        sx = scale[:, (0), (0)]
        ls = torch.pow(F.relu(torch.abs(sx) - self.scale_alpha), 2)
        sy = scale[:, (1), (1)]
        ly = torch.pow(F.relu(torch.abs(sy) - self.scale_alpha), 2)
        positive_loss = F.relu(self.positive_beta - sx) + F.relu(self.
            positive_beta - sy)
        loss = 0.1 * positive_loss + ls + ly
        return loss.sum().view(1)

    def anchor_loss(self, attention_region):
        distance = 0.5 * torch.pow(attention_region - self.anchor_box, 2).sum(1
            )
        return distance.sum().view(1)

    def forward(self, input, target):
        category, transform_box = input
        scores, index = category.max(1)
        bce_loss = self.bce(scores, target)
        regions = transform_box[:, :, :, 2:]
        region_loss = torch.cat([self.anchor_loss(i) for i in regions]).mean()
        scales = transform_box[:, :, :, :2]
        scale_loss = torch.cat([self.scale_loss(i) for i in scales]).mean()
        boundary = torch.abs(transform_box).sum(-1)
        boundary = torch.pow(F.relu(boundary - 1), 2)
        boundary_loss = boundary.view(boundary.size(0), -1).sum(-1).mean()
        return (bce_loss, bce_loss + 0.2 * region_loss + 0.05 * scale_loss +
            0.1 * boundary_loss)


def gram_matrix(feat):
    b, ch, h, w = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        ) + torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):

    def __init__(self, feature_encoder, feature_range=3):
        super(InpaintingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.feature_encoder = FeatureExtractor(feature_encoder, feature_range)

    def forward(self, raw_input, mask, output, origin):
        comp_img = mask * raw_input + (1 - mask) * output
        loss_validate = self.l1(mask * output, mask * origin)
        loss_hole = self.l1((1 - mask) * output, (1 - mask) * origin)
        loss_total_var = total_variation_loss(comp_img)
        feature_comp = self.feature_encoder(comp_img)
        feature_output = self.feature_encoder(output)
        feature_origin = self.feature_encoder(origin)
        loss_perceptual_1 = sum(map(lambda x, y: self.l1(x, y),
            feature_comp, feature_origin))
        loss_perceptual_2 = sum(map(lambda x, y: self.l1(x, y),
            feature_output, feature_origin))
        loss_perceptual = loss_perceptual_1 + loss_perceptual_2
        loss_style_1 = sum(map(lambda x, y: self.l1(gram_matrix(x),
            gram_matrix(y)), feature_output, feature_origin))
        loss_style_2 = sum(map(lambda x, y: self.l1(gram_matrix(x),
            gram_matrix(y)), feature_comp, feature_origin))
        loss_style = loss_style_1 + loss_style_2
        loss = (1.0 * loss_validate + 6.0 * loss_hole + 0.1 *
            loss_total_var + 0.05 * loss_perceptual + 120 * loss_style)
        return loss


class FeatureExtractor(nn.Module):

    def __init__(self, encoder, feature_range=3):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(*[encoder.features[i] for i in range(
            feature_range)])
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class DoubleAvdPool(nn.AvgPool2d):

    def __init__(self, kernel_size):
        super(DoubleAvdPool, self).__init__(kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, args):
        type(args)
        return tuple(map(lambda x: avg_pool2d(x, kernel_size=self.
            kernel_size), args))


class DoubleUpSample(nn.Module):

    def __init__(self, scale_factor, mode='nearest'):
        super(DoubleUpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, args):
        x, mask = args
        return self.upsample(x), self.upsample(mask)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yu45020_Text_Segmentation_Image_Inpainting(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DoubleAvdPool(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(tofp16(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(tofp32(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

