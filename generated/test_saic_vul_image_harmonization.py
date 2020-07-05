import sys
_module = sys.modules[__name__]
del sys
base = _module
compose = _module
hdataset = _module
transforms = _module
event_loop = _module
optimizer = _module
simple_trainer = _module
evaluation = _module
metrics = _module
predictor = _module
utils = _module
mconfigs = _module
backboned = _module
deeplab = _module
hrnet = _module
ih_model = _module
dih_model = _module
iseunet_v1 = _module
ssam_model = _module
initializer = _module
losses = _module
metrics = _module
basic_blocks = _module
conv_autoencoder = _module
deeplab_v3 = _module
hrnet_ocr = _module
ocr = _module
resnet = _module
resnetv1b = _module
unet = _module
modifiers = _module
ops = _module
syncbn = _module
modules = _module
functional = _module
_csrc = _module
nn = _module
syncbn = _module
exp = _module
log = _module
misc = _module
dih = _module
hrnet18_idih = _module
hrnet18_idih_no_mask = _module
hrnet18_sedih = _module
hrnet18_ssam = _module
improved_dih = _module
deeplab_idih = _module
hrnet18_issam = _module
hrnet18_v2p_idih = _module
hrnet18s_idih = _module
hrnet18s_issam = _module
hrnet18s_no_mask_idih = _module
hrnet18s_sedih = _module
hrnet18s_v2p_idih = _module
hrnet32_idih = _module
improved_sedih = _module
improved_ssam = _module
ssam = _module
evaluate_model = _module
evaluate_model_fg_ratios = _module
predict_for_dir = _module
train = _module

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


import logging


from copy import deepcopy


from collections import defaultdict


import torch


import numpy as np


from torch.utils.data import DataLoader


from torchvision.transforms import Normalize


from torch import nn as nn


import torch.nn as nn


from functools import partial


import torch.nn.functional as F


import math


import numbers


from torch import nn


import torch._utils


from torch.nn import functional as F


from torch.nn.parameter import Parameter


class _CustomDP(torch.nn.DataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class LRMult(object):

    def __init__(self, lr_mult=1.0):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult


class DeepLabBB(nn.Module):

    def __init__(self, pyramid_channels=256, deeplab_ch=256, backbone='resnet34', backbone_lr_mult=0.1):
        super(DeepLabBB, self).__init__()
        self.pyramid_on = pyramid_channels > 0
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        else:
            self.output_channels = [deeplab_ch]
        self.deeplab = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, project_dropout=0.2, norm_layer=nn.BatchNorm2d, backbone_norm_layer=nn.BatchNorm2d)
        self.deeplab.backbone.apply(LRMult(backbone_lr_mult))
        if self.pyramid_on:
            self.downsize = MaxPoolDownSize(deeplab_ch, pyramid_channels, pyramid_channels, 4)

    def forward(self, image, mask, mask_features):
        outputs = list(self.deeplab(image, mask_features))
        if self.pyramid_on:
            outputs = self.downsize(outputs[0])
        return outputs

    def load_pretrained_weights(self):
        self.deeplab.load_pretrained_weights()


class HRNetBB(nn.Module):

    def __init__(self, cat_outputs=True, pyramid_channels=256, pyramid_depth=4, width=18, ocr=64, small=True, lr_mult=0.1):
        super(HRNetBB, self).__init__()
        self.cat_outputs = cat_outputs
        self.ocr_on = ocr > 0 and cat_outputs
        self.pyramid_on = pyramid_channels > 0 and cat_outputs
        self.hrnet = HighResolutionNet(width, 2, ocr_width=ocr, small=small)
        self.hrnet.apply(LRMult(lr_mult))
        if self.ocr_on:
            self.hrnet.ocr_distri_head.apply(LRMult(1.0))
            self.hrnet.ocr_gather_head.apply(LRMult(1.0))
            self.hrnet.conv3x3_ocr.apply(LRMult(1.0))
        hrnet_cat_channels = [(width * 2 ** i) for i in range(4)]
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        elif self.ocr_on:
            self.output_channels = [ocr * 2]
        elif self.cat_outputs:
            self.output_channels = [sum(hrnet_cat_channels)]
        else:
            self.output_channels = hrnet_cat_channels
        if self.pyramid_on:
            downsize_in_channels = ocr * 2 if self.ocr_on else sum(hrnet_cat_channels)
            self.downsize = MaxPoolDownSize(downsize_in_channels, pyramid_channels, pyramid_channels, pyramid_depth)

    def forward(self, image, mask, mask_features):
        if not self.cat_outputs:
            return self.hrnet.compute_hrnet_feats(image, mask_features, return_list=True)
        outputs = list(self.hrnet(image, mask, mask_features))
        if self.pyramid_on:
            outputs = self.downsize(outputs[0])
        return outputs

    def load_pretrained_weights(self, pretrained_path):
        self.hrnet.load_pretrained_weights(pretrained_path)


class IHModelWithBackbone(nn.Module):

    def __init__(self, model, backbone, downsize_backbone_input=False, mask_fusion='sum', backbone_conv1_channels=64):
        """
        Creates image harmonization model supported by the features extracted from the pre-trained backbone.

        Parameters
        ----------
        model : nn.Module
            Image harmonization model takes image and mask as an input and handles features from the backbone network.
        backbone : nn.Module
            Backbone model accepts RGB image and returns a list of features.
        downsize_backbone_input : bool
            If the input image should be half-sized for the backbone.
        mask_fusion : str
            How to fuse the binary mask with the backbone input:
            'sum': apply convolution to the mask and sum it with the output of the first convolution in the backbone
            'rgb': concatenate the mask to the input image and translate it back to 3 channels with convolution
            otherwise: do not fuse mask with the backbone input
        backbone_conv1_channels : int
            If mask_fusion is 'sum', define the number of channels for the convolution applied to the mask.
        """
        super(IHModelWithBackbone, self).__init__()
        self.downsize_backbone_input = downsize_backbone_input
        self.mask_fusion = mask_fusion
        self.backbone = backbone
        self.model = model
        if mask_fusion == 'rgb':
            self.fusion = SimpleInputFusion()
        elif mask_fusion == 'sum':
            self.mask_conv = nn.Sequential(nn.Conv2d(1, backbone_conv1_channels, kernel_size=3, stride=2, padding=1, bias=True), ScaleLayer(init_value=0.1, lr_mult=1))

    def forward(self, image, mask):
        """
        Forward the backbone model and then the base model, supported by the backbone feature maps.
        Return model predictions.

        Parameters
        ----------
        image : torch.Tensor
            Input RGB image.
        mask : torch.Tensor
            Binary mask of the foreground region.

        Returns
        -------
        torch.Tensor
            Harmonized RGB image.
        """
        backbone_image = image
        backbone_mask = torch.cat((mask, 1.0 - mask), dim=1)
        if self.downsize_backbone_input:
            backbone_image = nn.functional.interpolate(backbone_image, scale_factor=0.5, mode='bilinear', align_corners=True)
            backbone_mask = nn.functional.interpolate(backbone_mask, backbone_image.size()[2:], mode='bilinear', align_corners=True)
        backbone_image = self.fusion(backbone_image, backbone_mask[:, :1]) if self.mask_fusion == 'rgb' else backbone_image
        backbone_mask_features = self.mask_conv(backbone_mask[:, :1]) if self.mask_fusion == 'sum' else None
        backbone_features = self.backbone(backbone_image, backbone_mask, backbone_mask_features)
        output = self.model(image, mask, backbone_features)
        return output


class DeepImageHarmonization(nn.Module):

    def __init__(self, depth, norm_layer=nn.BatchNorm2d, batchnorm_from=0, attend_from=-1, image_fusion=False, ch=64, max_channels=512, backbone_from=-1, backbone_channels=None, backbone_mode=''):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(depth, ch, norm_layer, batchnorm_from, max_channels, backbone_from, backbone_channels, backbone_mode)
        self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {'images': output}


class ISEUNetV1(nn.Module):

    def __init__(self, depth, norm_layer=nn.BatchNorm2d, batchnorm_from=2, attend_from=3, image_fusion=False, ch=64, max_channels=512, backbone_from=-1, backbone_channels=None, backbone_mode=''):
        super(ISEUNetV1, self).__init__()
        self.depth = depth
        self.encoder = UNetEncoder(depth, ch, norm_layer, batchnorm_from, max_channels, backbone_from, backbone_channels, backbone_mode)
        self.decoder = UNetDecoder(depth, self.encoder.block_channels, norm_layer, attention_layer=MaskedChannelAttention, attend_from=attend_from, image_fusion=image_fusion)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {'images': output}


class SSAMImageHarmonization(nn.Module):

    def __init__(self, depth, norm_layer=nn.BatchNorm2d, batchnorm_from=2, attend_from=3, attention_mid_k=2.0, image_fusion=False, ch=64, max_channels=512, backbone_from=-1, backbone_channels=None, backbone_mode=''):
        super(SSAMImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = UNetEncoder(depth, ch, norm_layer, batchnorm_from, max_channels, backbone_from, backbone_channels, backbone_mode)
        self.decoder = UNetDecoder(depth, self.encoder.block_channels, norm_layer, attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k), attend_from=attend_from, image_fusion=image_fusion)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {'images': output}


class SpatialSeparatedAttention(nn.Module):

    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)
        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(ConvBlock(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation, bias=False), ConvBlock(mid_channels, in_channels, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation, bias=False))
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(nn.functional.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=True))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output


class Loss(nn.Module):

    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class ConvHead(nn.Module):

    def __init__(self, out_channels, in_channels=32, num_layers=1, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []
        for i in range(num_layers):
            convhead.extend([nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding), nn.ReLU(), norm_layer(in_channels) if norm_layer is not None else nn.Identity()])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class SepConvHead(nn.Module):

    def __init__(self, num_outputs, in_channels, mid_channels, num_layers=1, kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0, norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()
        sepconvhead = []
        for i in range(num_layers):
            sepconvhead.append(SeparableConv2d(in_channels=in_channels if i == 0 else mid_channels, out_channels=mid_channels, dw_kernel=kernel_size, dw_padding=padding, norm_layer=norm_layer, activation='relu'))
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))
        sepconvhead.append(nn.Conv2d(in_channels=mid_channels, out_channels=num_outputs, kernel_size=1, padding=0))
        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]
        return self.layers(x)


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f'Unknown activation type {activation}')
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f'Unknown activation type {activation}')


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1, activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = select_activation_function(activation)
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride, padding=dw_padding, bias=use_bias, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias), norm_layer(out_channels) if norm_layer is not None else nn.Identity(), _activation())

    def forward(self, x):
        return self.body(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=nn.BatchNorm2d, activation=nn.ELU, bias=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), norm_layer(out_channels) if norm_layer is not None else nn.Identity(), activation())

    def forward(self, x):
        return self.block(x)


class GaussianSmoothing(nn.Module):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    Apply gaussian smoothing on a tensor (1d, 2d, 3d).
    Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, padding=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1.0
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.0
            kernel *= torch.exp(-((grid - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** 0.5)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = torch.repeat_interleave(kernel, channels, 0)
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, padding=self.padding, groups=self.groups)


class MaxPoolDownSize(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super(MaxPoolDownSize, self).__init__()
        self.depth = depth
        self.reduce_conv = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.convs = nn.ModuleList([ConvBlock(mid_channels, out_channels, kernel_size=3, stride=1, padding=1) for conv_i in range(depth)])
        self.pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        outputs = []
        output = self.reduce_conv(x)
        for conv_i, conv in enumerate(self.convs):
            output = output if conv_i == 0 else self.pool2d(output)
            outputs.append(conv(output))
        return outputs


class ConvEncoder(nn.Module):

    def __init__(self, depth, ch, norm_layer, batchnorm_from, max_channels, backbone_from, backbone_channels=None, backbone_mode=''):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]
        in_channels = 4
        out_channels = ch
        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]
        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels
            self.blocks_connected[f'block{block_i}'] = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None, padding=int(block_i < depth - 1))
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]
        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]
        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]
        return outputs[::-1]


class DeconvDecoder(nn.Module):

    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(in_channels, out_channels, norm_layer=norm_layer, padding=0 if d == 0 else 1, with_se=0 <= attend_from <= d))
            in_channels = out_channels
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)
        return output


class SEDeconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=nn.BatchNorm2d, activation=nn.ELU, with_se=False):
        super(SEDeconvBlock, self).__init__()
        self.with_se = with_se
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), norm_layer(out_channels) if norm_layer is not None else nn.Identity(), activation())
        if self.with_se:
            self.se = MaskedChannelAttention(out_channels)

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_se:
            out = self.se(out, mask)
        return out


class DeepLabV3Plus(nn.Module):

    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d, backbone_norm_layer=None, ch=256, project_dropout=0.5, inference_mode=False, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer
        self.backbone_name = backbone
        self.norm_layer = norm_layer
        self.backbone_norm_layer = backbone_norm_layer
        self.inference_mode = False
        self.ch = ch
        self.aspp_in_channels = 2048
        self.skip_project_in_channels = 256
        self._kwargs = kwargs
        if backbone == 'resnet34':
            self.aspp_in_channels = 512
            self.skip_project_in_channels = 64
        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, norm_layer=self.backbone_norm_layer, **kwargs)
        self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch, norm_layer=self.norm_layer)
        self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
        self.aspp = _ASPP(in_channels=self.aspp_in_channels, atrous_rates=[12, 24, 36], out_channels=ch, project_dropout=project_dropout, norm_layer=self.norm_layer)
        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, norm_layer=self.backbone_norm_layer, **self._kwargs)
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict)
        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, x, mask_features=None):
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(torch.no_grad())
            c1, _, c3, c4 = self.backbone(x, mask_features)
            c1 = self.skip_project(c1)
            x = self.aspp(c4)
            x = F.interpolate(x, c1.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, c1), dim=1)
            x = self.head(x)
        return x,


class _SkipProject(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_SkipProject, self).__init__()
        _activation = select_activation_function('relu')
        self.skip_project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), norm_layer(out_channels), _activation())

    def forward(self, x):
        return self.skip_project(x)


class _DeepLabHead(nn.Module):

    def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.block = nn.Sequential(SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        return self.block(x)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False), norm_layer(out_channels), nn.ReLU())
    return block


class _ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates, out_channels=256, project_dropout=0.5, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        b0 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())
        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
        self.concurent = nn.ModuleList([b0, b1, b2, b3, b4])
        project = [nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU()]
        if project_dropout > 0:
            project.append(nn.Dropout(project_dropout))
        self.project = nn.Sequential(*project)

    def forward(self, x):
        x = torch.cat([block(x) for block in self.concurent], dim=1)
        return self.project(x)


class _AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, x.size()[2:], mode='bilinear', align_corners=True)


relu_inplace = True


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.norm_layer = norm_layer
        self.align_corners = align_corners
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(num_channels[branch_index] * block.expansion))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample=downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i], kernel_size=1, bias=False), self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3), nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


LOGGER_NAME = 'root'


logger = logging.getLogger(LOGGER_NAME)


class HighResolutionNet(nn.Module):

    def __init__(self, width, num_classes, ocr_width=256, small=False, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionNet, self).__init__()
        self.norm_layer = norm_layer
        self.width = width
        self.ocr_width = ocr_width
        self.ocr_on = ocr_width > 0
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=relu_inplace)
        num_blocks = 2 if small else 4
        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleneckV1b, 64, stage1_num_channels, blocks=num_blocks)
        stage1_out_channel = BottleneckV1b.expansion * stage1_num_channels
        self.stage2_num_branches = 2
        num_channels = [width, 2 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1, num_branches=self.stage2_num_branches, num_blocks=2 * [num_blocks], num_channels=num_channels)
        self.stage3_num_branches = 3
        num_channels = [width, 2 * width, 4 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=3 if small else 4, num_branches=self.stage3_num_branches, num_blocks=3 * [num_blocks], num_channels=num_channels)
        self.stage4_num_branches = 4
        num_channels = [width, 2 * width, 4 * width, 8 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3, num_branches=self.stage4_num_branches, num_blocks=4 * [num_blocks], num_channels=num_channels)
        if self.ocr_on:
            last_inp_channels = np.int(np.sum(pre_stage_channels))
            ocr_mid_channels = 2 * ocr_width
            ocr_key_channels = ocr_width
            self.conv3x3_ocr = nn.Sequential(nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1), norm_layer(ocr_mid_channels), nn.ReLU(inplace=relu_inplace))
            self.ocr_gather_head = SpatialGather_Module(num_classes)
            self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels, key_channels=ocr_key_channels, out_channels=ocr_mid_channels, scale=1, dropout=0.05, norm_layer=norm_layer, align_corners=align_corners)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(num_channels_cur_layer[i]), nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(outchannels), nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels, num_modules, num_branches, num_blocks, num_channels, fuse_method='SUM', multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output, norm_layer=self.norm_layer, align_corners=self.align_corners))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, mask=None, additional_features=None):
        hrnet_feats = self.compute_hrnet_feats(x, additional_features)
        if not self.ocr_on:
            return hrnet_feats,
        ocr_feats = self.conv3x3_ocr(hrnet_feats)
        mask = nn.functional.interpolate(mask, size=ocr_feats.size()[2:], mode='bilinear', align_corners=True)
        context = self.ocr_gather_head(ocr_feats, mask)
        ocr_feats = self.ocr_distri_head(ocr_feats, context)
        return ocr_feats,

    def compute_hrnet_feats(self, x, additional_features, return_list=False):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        if return_list:
            return x
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        return torch.cat([x[0], x1, x2, x3], 1)

    def compute_pre_stage_features(self, x, additional_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + additional_features
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)

    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()
        if not os.path.exists(pretrained_path):
            None
            None
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}
        params_count = len(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        logger.info(f'Loaded {len(pretrained_dict)} of {params_count} pretrained parameters for HRNet')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3).contiguous()
        return ocr_context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, norm_layer, align_corners)
        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.Sequential(norm_layer(out_channels), nn.ReLU(inplace=True)), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ObjectAttentionBlock2D(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_object = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_down = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_up = nn.Sequential(nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.in_channels), nn.ReLU(inplace=True)))

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context


GLUON_RESNET_TORCH_HUB = 'rwightman/pytorch-pretrained-gluonresnet'


def _safe_state_dict_filtering(orig_dict, model_dict_keys):
    filtered_orig_dict = {}
    for k, v in orig_dict.items():
        if k in model_dict_keys:
            filtered_orig_dict[k] = v
        else:
            print(f'[ERROR] Failed to load <{k}> in backbone')
    return filtered_orig_dict


def resnet101_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet101_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet152_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet34_v1b', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet50_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


class ResNetBackbone(torch.nn.Module):

    def __init__(self, backbone='resnet50', pretrained_base=True, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()
        if backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x, mask_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if mask_features is not None:
            x = x + mask_features
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNetV1b(nn.Module):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm2d`)
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, classes=1000, dilated=True, deep_stem=False, stem_width=32, avg_down=False, final_drop=0.0, norm_layer=nn.BatchNorm2d):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, 2 * stem_width, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, avg_down=avg_down, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, avg_down=False, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = []
            if avg_down:
                if dilation == 1:
                    downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    downsample.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                downsample.extend([nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=1, bias=False), norm_layer(planes * block.expansion)])
                downsample = nn.Sequential(*downsample)
            else:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)
        return x


class UNetEncoder(nn.Module):

    def __init__(self, depth, ch, norm_layer, batchnorm_from, max_channels, backbone_from, backbone_channels=None, backbone_mode=''):
        super(UNetEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        self.block_channels = []
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]
        relu = partial(nn.ReLU, inplace=True)
        in_channels = 4
        out_channels = ch
        self.block0 = UNetDownBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None, activation=relu, pool=True, padding=1)
        self.block_channels.append(out_channels)
        in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
        self.block1 = UNetDownBlock(in_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None, activation=relu, pool=True, padding=1)
        self.block_channels.append(out_channels)
        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels
            self.blocks_connected[f'block{block_i}'] = UNetDownBlock(in_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None, activation=relu, padding=1, pool=block_i < depth - 1)
            self.block_channels.append(out_channels)

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]
        outputs = []
        block_input = x
        output, block_input = self.block0(block_input)
        outputs.append(output)
        output, block_input = self.block1(block_input)
        outputs.append(output)
        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                block_input = connector(block_input, stage_features)
            output, block_input = block(block_input)
            outputs.append(output)
        return outputs[::-1]


class UNetDecoder(nn.Module):

    def __init__(self, depth, encoder_blocks_channels, norm_layer, attention_layer=None, attend_from=3, image_fusion=False):
        super(UNetDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(in_channels, out_channels, out_channels, norm_layer=norm_layer, activation=partial(nn.ReLU, inplace=True), padding=1, attention_layer=stage_attention_layer))
            in_channels = out_channels
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, input_image, mask):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask)
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * input_image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)
        return output


class UNetDownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = UNetDoubleConv(in_channels, out_channels, norm_layer=norm_layer, activation=activation, padding=padding)
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)


class UNetUpBlock(nn.Module):

    def __init__(self, in_channels_decoder, in_channels_encoder, out_channels, norm_layer, activation, padding, attention_layer):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), ConvBlock(in_channels_decoder, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=None, activation=activation))
        self.convs = UNetDoubleConv(in_channels_encoder + out_channels, out_channels, norm_layer=norm_layer, activation=activation, padding=padding)
        if attention_layer is not None:
            self.attention = attention_layer(in_channels_encoder + out_channels, norm_layer, activation)
        else:
            self.attention = None

    def forward(self, x, encoder_out, mask=None):
        upsample_x = self.upconv(x)
        x_cat_encoder = torch.cat([encoder_out, upsample_x], dim=1)
        if self.attention is not None:
            x_cat_encoder = self.attention(x_cat_encoder, mask)
        return self.convs(x_cat_encoder)


class UNetDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super(UNetDoubleConv, self).__init__()
        self.block = nn.Sequential(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, norm_layer=norm_layer, activation=activation), ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, norm_layer=norm_layer, activation=activation))

    def forward(self, x):
        return self.block(x)


class SimpleInputFusion(nn.Module):

    def __init__(self, add_ch=1, rgb_ch=3, ch=8, norm_layer=nn.BatchNorm2d):
        super(SimpleInputFusion, self).__init__()
        self.fusion_conv = nn.Sequential(nn.Conv2d(in_channels=add_ch + rgb_ch, out_channels=ch, kernel_size=1), nn.LeakyReLU(negative_slope=0.2), norm_layer(ch), nn.Conv2d(in_channels=ch, out_channels=rgb_ch, kernel_size=1))

    def forward(self, image, additional_input):
        return self.fusion_conv(torch.cat((image, additional_input), dim=1))


class ChannelAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)])
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(nn.Linear(len(self.global_pools) * in_channels, intermediate_channels_count), nn.ReLU(), nn.Linear(intermediate_channels_count, in_channels), nn.Sigmoid())

    def forward(self, x):
        pooled_x = []
        for global_pool in self.global_pools:
            pooled_x.append(global_pool(x))
        pooled_x = torch.cat(pooled_x, dim=1).flatten(start_dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x


class MaskedChannelAttention(nn.Module):

    def __init__(self, in_channels, *args, **kwargs):
        super(MaskedChannelAttention, self).__init__()
        self.global_max_pool = MaskedGlobalMaxPool2d()
        self.global_avg_pool = FastGlobalAvgPool2d()
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(nn.Linear(3 * in_channels, intermediate_channels_count), nn.ReLU(inplace=True), nn.Linear(intermediate_channels_count, in_channels), nn.Sigmoid())

    def forward(self, x, mask):
        if mask.shape[2:] != x.shape[:2]:
            mask = nn.functional.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=True)
        pooled_x = torch.cat([self.global_max_pool(x, mask), self.global_avg_pool(x)], dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x


class MaskedGlobalMaxPool2d(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_max_pool = FastGlobalMaxPool2d()

    def forward(self, x, mask):
        return torch.cat((self.global_max_pool(x * mask), self.global_max_pool(x * (1.0 - mask))), dim=1)


class FastGlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(FastGlobalAvgPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).mean(dim=2)


class FastGlobalMaxPool2d(nn.Module):

    def __init__(self):
        super(FastGlobalMaxPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).max(dim=2)[0]


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(torch.full((1,), init_value / lr_mult, dtype=torch.float32))

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class FeaturesConnector(nn.Module):

    def __init__(self, mode, in_channels, feature_channels, out_channels):
        super(FeaturesConnector, self).__init__()
        self.mode = mode if feature_channels else ''
        if self.mode == 'catc':
            self.reduce_conv = nn.Conv2d(in_channels + feature_channels, out_channels, kernel_size=1)
        elif self.mode == 'sum':
            self.reduce_conv = nn.Conv2d(feature_channels, out_channels, kernel_size=1)
        self.output_channels = out_channels if self.mode != 'cat' else in_channels + feature_channels

    def forward(self, x, features):
        if self.mode == 'cat':
            return torch.cat((x, features), 1)
        if self.mode == 'catc':
            return self.reduce_conv(torch.cat((x, features), 1))
        if self.mode == 'sum':
            return self.reduce_conv(features) + x
        return x

    def extra_repr(self):
        return self.mode


class _BatchNorm(nn.Module):
    """
    Customized BatchNorm from nn.BatchNorm
    >> added freeze attribute to enable bn freeze.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.freezed = False
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        compute_stats = not self.freezed and self.training and self.track_running_stats
        ret = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, compute_stats, self.momentum, self.eps)
        return ret

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlockV1b,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelAttention,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepImageHarmonization,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (FastGlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastGlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeaturesConnector,
     lambda: ([], {'mode': 4, 'in_channels': 4, 'feature_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianSmoothing,
     lambda: ([], {'channels': 4, 'kernel_size': 4, 'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HRNetBB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 64, 32, 32]), torch.rand([4, 64, 32, 32])], {}),
     False),
    (ISEUNetV1,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (MaskedChannelAttention,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedGlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolDownSize,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ObjectAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SEDeconvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSAMImageHarmonization,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (ScaleLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleInputFusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (SpatialGather_Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialOCR_Module,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNetDoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer, 'activation': _mock_layer, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNetDownBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer, 'activation': _mock_layer, 'pool': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_AsppPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_CustomDP,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (_DeepLabHead,
     lambda: ([], {'out_channels': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_SkipProject,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_saic_vul_image_harmonization(_paritybench_base):
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

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

