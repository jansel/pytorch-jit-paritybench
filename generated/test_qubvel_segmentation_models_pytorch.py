import sys
_module = sys.modules[__name__]
del sys
master = _module
generate_table = _module
segmentation_models_pytorch = _module
__version__ = _module
base = _module
heads = _module
initialization = _module
model = _module
modules = _module
deeplabv3 = _module
decoder = _module
model = _module
encoders = _module
_base = _module
_preprocessing = _module
_utils = _module
densenet = _module
dpn = _module
efficientnet = _module
inceptionresnetv2 = _module
inceptionv4 = _module
mobilenet = _module
resnet = _module
senet = _module
timm_efficientnet = _module
vgg = _module
xception = _module
fpn = _module
decoder = _module
model = _module
linknet = _module
decoder = _module
model = _module
pan = _module
decoder = _module
model = _module
pspnet = _module
decoder = _module
model = _module
unet = _module
decoder = _module
model = _module
utils = _module
base = _module
functional = _module
losses = _module
meter = _module
metrics = _module
train = _module
setup = _module
test_models = _module
test_preprocessing = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torch


from torch import nn


from torch.nn import functional as F


from typing import Optional


import functools


import torch.utils.model_zoo as model_zoo


from typing import List


from collections import OrderedDict


import re


from torchvision.models.densenet import DenseNet


import torch.nn.functional as F


import torchvision


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.vgg import VGG


from torchvision.models.vgg import make_layers


from typing import Union


import numpy as np


class Activation(nn.Module):

    def __init__(self, activation):
        super().__init__()
        if activation == None or activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softmax2d':
            self.activation = functools.partial(torch.softmax, dim=1)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling='avg', dropout=0.2, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x


class Conv2dReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        if use_batchnorm == 'inplace' and InPlaceABN is None:
            raise RuntimeError("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. " + 'To install see: https://github.com/mapillary/inplace_abn')
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        if use_batchnorm == 'inplace':
            bn = InPlaceABN(out_channels, activation='leaky_relu', activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != 'inplace':
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, in_channels, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=dim)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError('Attention {} is not implemented'.format(name))

    def forward(self, x):
        return self.attention(x)


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        dephtwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)


class ASPPSeparableConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv
        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Decoder(nn.Sequential):

    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(ASPP(in_channels, out_channels, atrous_rates), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):

    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError('Output stride should be 8 or 16, got {}.'.format(output_stride))
        self.out_channels = out_channels
        self.output_stride = output_stride
        self.aspp = nn.Sequential(ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True), SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48
        self.block1 = nn.Sequential(nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(highres_out_channels), nn.ReLU())
        self.block2 = nn.Sequential(SeparableConv2d(highres_out_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


encoders = {}


def get_encoder(name, in_channels=3, depth=5, weights=None):
    Encoder = encoders[name]['encoder']
    params = encoders[name]['params']
    params.update(depth=depth)
    encoder = Encoder(**params)
    if weights is not None:
        settings = encoders[name]['pretrained_settings'][weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))
    encoder.set_in_channels(in_channels)
    return encoder


class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implemetation from "Rethinking Atrous Convolution for Semantic Image Segmentation"
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: a number of convolution filters in ASPP module (default 256).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 8 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**
    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587
    """

    def __init__(self, encoder_name: str='resnet34', encoder_depth: int=5, encoder_weights: Optional[str]='imagenet', decoder_channels: int=256, in_channels: int=3, classes: int=1, activation: Optional[str]=None, upsampling: int=8, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])
        self.decoder = DeepLabV3Decoder(in_channels=self.encoder.out_channels[-1], out_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3Plus_ implemetation from "Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation"
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_output_stride: downsampling factor for deepest encoder features (see original paper for explanation)
        decoder_atrous_rates: dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: a number of convolution filters in ASPP module (default 256).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 8 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    .. _DeeplabV3Plus:
        https://arxiv.org/abs/1802.02611v3
    """

    def __init__(self, encoder_name: str='resnet34', encoder_depth: int=5, encoder_weights: Optional[str]='imagenet', encoder_output_stride: int=16, decoder_channels: int=256, decoder_atrous_rates: tuple=(12, 24, 36), in_channels: int=3, classes: int=1, activation: Optional[str]=None, upsampling: int=4, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        if encoder_output_stride == 8:
            self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])
        elif encoder_output_stride == 16:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])
        else:
            raise ValueError('Encoder output stride should be 8 or 16, got {}'.format(encoder_output_stride))
        self.decoder = DeepLabV3PlusDecoder(encoder_channels=self.encoder.out_channels, out_channels=decoder_channels, atrous_rates=decoder_atrous_rates, output_stride=encoder_output_stride)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class TransitionWithSkip(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        for module in self.module:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip


class Conv3x3GNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False), nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):

    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):

    def __init__(self, policy):
        super().__init__()
        if policy not in ['add', 'cat']:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):

    def __init__(self, encoder_channels, encoder_depth=5, pyramid_channels=256, segmentation_channels=128, dropout=0.2, merge_policy='add'):
        super().__init__()
        self.out_channels = segmentation_channels if merge_policy == 'add' else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError('Encoder depth for FPN decoder cannot be less than 3, got {}.'.format(encoder_depth))
        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.seg_blocks = nn.ModuleList([SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [3, 2, 1, 0]])
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x


class FPN(SegmentationModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
        decoder_dropout: spatial dropout rate in range (0, 1).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(self, encoder_name: str='resnet34', encoder_depth: int=5, encoder_weights: Optional[str]='imagenet', decoder_pyramid_channels: int=256, decoder_segmentation_channels: int=128, decoder_merge_policy: str='add', decoder_dropout: float=0.2, in_channels: int=3, classes: int=1, activation: Optional[str]=None, upsampling: int=4, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = FPNDecoder(encoder_channels=self.encoder.out_channels, encoder_depth=encoder_depth, pyramid_channels=decoder_pyramid_channels, segmentation_channels=decoder_segmentation_channels, dropout=decoder_dropout, merge_policy=decoder_merge_policy)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'fpn-{}'.format(encoder_name)
        self.initialize()


class TransposeX2(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        self.conv1 = md.Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class LinknetDecoder(nn.Module):

    def __init__(self, encoder_channels, prefinal_channels=32, n_blocks=5, use_batchnorm=True):
        super().__init__()
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        channels = list(encoder_channels) + [prefinal_channels]
        self.blocks = nn.ModuleList([DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm) for i in range(n_blocks)])

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        x = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(self, encoder_name: str='resnet34', encoder_depth: int=5, encoder_weights: Optional[str]='imagenet', decoder_use_batchnorm: bool=True, in_channels: int=3, classes: int=1, activation: Optional[Union[str, callable]]=None, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = LinknetDecoder(encoder_channels=self.encoder.out_channels, n_blocks=encoder_depth, prefinal_channels=32, use_batchnorm=decoder_use_batchnorm)
        self.segmentation_head = SegmentationHead(in_channels=32, out_channels=classes, activation=activation, kernel_size=1)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'link-{}'.format(encoder_name)
        self.initialize()


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=True, add_relu: bool=True, interpolate: bool=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPABlock(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_mode='bilinear'):
        super(FPABlock, self).__init__()
        self.upscale_mode = upscale_mode
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = False
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=in_channels, out_channels=1, kernel_size=7, stride=1, padding=3))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)
        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)
        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)
        x = torch.mul(x, mid)
        x = x + b1
        return x


class GAUBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upscale_mode: str='bilinear'):
        super(GAUBlock, self).__init__()
        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == 'bilinear' else None
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=1, add_relu=False), nn.Sigmoid())
        self.conv2 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PANDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, upscale_mode: str='bilinear'):
        super().__init__()
        self.fpa = FPABlock(in_channels=encoder_channels[-1], out_channels=decoder_channels)
        self.gau3 = GAUBlock(in_channels=encoder_channels[-2], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau2 = GAUBlock(in_channels=encoder_channels[-3], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau1 = GAUBlock(in_channels=encoder_channels[-4], out_channels=decoder_channels, upscale_mode=upscale_mode)

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)
        x4 = self.gau3(features[-2], x5)
        x3 = self.gau2(features[-3], x4)
        x2 = self.gau1(features[-4], x3)
        return x2


class PAN(SegmentationModel):
    """ Implementation of _PAN (Pyramid Attention Network).
    Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
    and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1


    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_dilation: Flag to use dilation in encoder last layer.
            Doesn't work with [``*ception*``, ``vgg*``, ``densenet*``] backbones, default is True.
        decoder_channels: Number of ``Conv2D`` layer filters in decoder blocks
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)

        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **PAN**

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    """

    def __init__(self, encoder_name: str='resnet34', encoder_weights: str='imagenet', encoder_dilation: bool=True, decoder_channels: int=32, in_channels: int=3, classes: int=1, activation: Optional[Union[str, callable]]=None, upsampling: int=4, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        if encoder_dilation:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])
        self.decoder = PANDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels, out_channels=classes, activation=activation, kernel_size=3, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'pan-{}'.format(encoder_name)
        self.initialize()


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)), modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()
        self.blocks = nn.ModuleList([PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):

    def __init__(self, encoder_channels, use_batchnorm=True, out_channels=512, dropout=0.2):
        super().__init__()
        self.psp = PSPModule(in_channels=encoder_channels[-1], sizes=(1, 2, 3, 6), use_bathcnorm=use_batchnorm)
        self.conv = modules.Conv2dReLU(in_channels=encoder_channels[-1] * 2, out_channels=out_channels, kernel_size=1, use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class PSPNet(SegmentationModel):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        psp_out_channels: number of filters in PSP block.
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        psp_dropout: spatial dropout rate between 0 and 1.
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 8 for depth=3 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf
    """

    def __init__(self, encoder_name: str='resnet34', encoder_weights: Optional[str]='imagenet', encoder_depth: int=3, psp_out_channels: int=512, psp_use_batchnorm: bool=True, psp_dropout: float=0.2, in_channels: int=3, classes: int=1, activation: Optional[Union[str, callable]]=None, upsampling: int=8, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = PSPDecoder(encoder_channels=self.encoder.out_channels, use_batchnorm=psp_use_batchnorm, out_channels=psp_out_channels, dropout=psp_dropout)
        self.segmentation_head = SegmentationHead(in_channels=psp_out_channels, out_channels=classes, kernel_size=3, activation=activation, upsampling=upsampling)
        if aux_params:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'psp-{}'.format(encoder_name)
        self.initialize()


class CenterBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True, attention_type=None, center=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError('Model depth is {}, but you provide `decoder_channels` for {} blocks.'.format(n_blocks, len(decoder_channels)))
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(self, encoder_name: str='resnet34', encoder_depth: int=5, encoder_weights: str='imagenet', decoder_use_batchnorm: bool=True, decoder_channels: List[int]=(256, 128, 64, 32, 16), decoder_attention_type: Optional[str]=None, in_channels: int=3, classes: int=1, activation: Optional[Union[str, callable]]=None, aux_params: Optional[dict]=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm, center=True if encoder_name.startswith('vgg') else False, attention_type=decoder_attention_type)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'u-{}'.format(encoder_name)
        self.initialize()


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
            return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass


class Loss(BaseObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class JaccardLoss(base.Loss):

    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels)


class DiceLoss(base.Loss):

    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(y_pr, y_gt, beta=self.beta, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels)


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPSeparableConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClassificationHead,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FPABlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GAUBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSPBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'pool_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSPModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SCSEModule,
     lambda: ([], {'in_channels': 64}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     True),
    (SegmentationHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransposeX2,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_qubvel_segmentation_models_pytorch(_paritybench_base):
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

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

