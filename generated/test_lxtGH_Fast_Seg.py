import sys
_module = sys.modules[__name__]
del sys
libs = _module
core = _module
loss = _module
operators = _module
datasets = _module
camvid = _module
cityscapes = _module
mapillary = _module
BiSegNet = _module
DFANet = _module
DFSegNet = _module
ESPNet = _module
FastSCNN = _module
ICNet = _module
MSFNet = _module
PSPNet = _module
SwiftNet = _module
models = _module
backbone = _module
dfnet = _module
resnet = _module
xception = _module
utils = _module
image_utils = _module
logger = _module
tools = _module
prediction_test_different_size = _module
train_distribute = _module
val = _module

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


import torch.nn as nn


import torch


import torch.nn.functional as F


from torch.nn import BatchNorm2d


from torch.nn import functional as F


import torch.utils.model_zoo as model_zoo


import math


import numpy as np


import random


import collections


from scipy import ndimage


from torch.utils import data


from math import ceil


class OhemCrossEntropy2dTensor(nn.Module):

    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=
        0.6, min_kept=256, down_ratio=1, use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            None
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.
                long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


class CriterionDSN(nn.CrossEntropyLoss):

    def __init__(self, ignore_index=255, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, preds, target):
        scale_pred = preds[0]
        loss1 = super(CriterionDSN, self).forward(scale_pred, target)
        scale_pred = preds[1]
        loss2 = super(CriterionDSN, self).forward(scale_pred, target)
        return loss1 + loss2 * 0.4


class CriterionOhemDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the models.
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000,
        reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=
            thresh, min_kept=min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=
            ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)
        return loss1 + loss2 * 0.4


class CriterionICNet(nn.Module):
    """
    ICNet loss
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000,
        reduce=True):
        super(CriterionICNet, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=
            thresh, min_kept=min_kept)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss2 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[2], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss3 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[3], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss4 = self.criterion1(scale_pred, target)
        return loss1 + 0.4 * loss2 + 0.4 * loss3 + 0.4 * loss4


class CriterionDFANet(nn.Module):
    """
    ICNet loss
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000,
        reduce=True):
        super(CriterionDFANet, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=
            thresh, min_kept=min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=
            ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss2 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[2], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss3 = self.criterion1(scale_pred, target)
        return loss1 + 0.4 * loss2 + 0.4 * loss3


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)
        return inputs


class SELayer(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_planes, out_planes //
            reduction), nn.ReLU(inplace=True), nn.Linear(out_planes //
            reduction, out_planes), nn.Sigmoid())
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0,
        dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
        bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
            stride=stride, padding=pad, dilation=dilation, groups=groups,
            bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, 0, dilation, groups=in_channels, bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1
            )
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512,
        dilations=(12, 24, 36), norm_layer=nn.BatchNorm2d):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(
            features, inner_features, kernel_size=1, padding=0, dilation=1,
            bias=False), norm_layer(inner_features), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features,
            kernel_size=1, padding=0, dilation=1, bias=False), norm_layer(
            inner_features), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=False), norm_layer(inner_features), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=False), norm_layer(inner_features), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=False), norm_layer(inner_features), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(inner_features * 5,
            out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features), nn.ReLU(), nn.Dropout2d(0.1))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear',
            align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        bottle = self.bottleneck(out)
        return bottle


class A2Block(nn.Module):
    """
        Implementation of A2Block(NIPS 2018)
    """

    def __init__(self, inplane, plane):
        super(A2Block, self).__init__()
        self.down = nn.Conv2d(inplane, plane, 1)
        self.up = nn.Conv2d(plane, inplane, 1)
        self.gather_down = nn.Conv2d(inplane, plane, 1)
        self.distribue_down = nn.Conv2d(inplane, plane, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, w = A.size()
        A = A.view(b, c, -1)
        B = B.view(b, c, -1)
        B = self.softmax(B)
        B = B.permute(0, 2, 1)
        G = torch.bmm(A, B)
        C = self.distribue_down(res)
        C = C.view(b, c, -1)
        C = self.softmax(C)
        C = C.permute(0, 2, 1)
        atten = torch.bmm(C, G)
        atten = atten.permute(0, 2, 1).view(b, c, h, -1)
        atten = self.up(atten)
        out = res + atten
        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6),
        norm_layer=BatchNorm2d):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features,
            out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(features + len(sizes) *
            out_features, out_features, kernel_size=1, padding=1, dilation=
            1, bias=False), norm_layer(out_features), nn.ReLU(), nn.
            Dropout2d(0.1))

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode=
            'bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class AttentionRefinement(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bn=
            True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0, has_bn=True,
            norm_layer=norm_layer, has_relu=False, has_bias=False), nn.
            Sigmoid())

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se
        return fm


class FeatureFusion(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=1, norm_layer=nn.
        BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0, has_bn=
            True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0, has_bn
            =False, norm_layer=norm_layer, has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0, has_bn
            =False, norm_layer=norm_layer, has_relu=False, has_bias=False),
            nn.Sigmoid())

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class SpatialPath(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
            has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
            has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
            has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
            has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output


class BiSeNetHead(nn.Module):

    def __init__(self, in_planes, out_planes, scale, is_aux=False,
        norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1, has_bn=True,
                norm_layer=norm_layer, has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1, has_bn=True,
                norm_layer=norm_layer, has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1, stride
                =1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale, mode=
                'bilinear', align_corners=True)
        return output


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()
    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(', '.join('{}'.
            format(k) for k in missing_keys)))
    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(', '.join('{}'.
            format(k) for k in unexpected_keys)))
    del state_dict
    t_end = time.time()
    print('Load model, Time usage:\n\tIO: {}, initialize parameters: {}'.
        format(t_ioend - t_start, t_end - t_ioend))
    return model


def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class BiSeNet(nn.Module):

    def __init__(self, out_planes, is_training=False, pretrained_model=None,
        norm_layer=nn.BatchNorm2d):
        super(BiSeNet, self).__init__()
        self.backbone = resnet18(pretrained_model, norm_layer=norm_layer,
            bn_eps=1e-05, bn_momentum=0.1, deep_stem=True, stem_width=64)
        self.business_layer = []
        self.is_training = is_training
        self.spatial_path = SpatialPath(3, 128, norm_layer)
        conv_channel = 128
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0, has_bn=True, has_relu=
            True, has_bias=False, norm_layer=norm_layer))
        arms = [AttentionRefinement(512, conv_channel, norm_layer),
            AttentionRefinement(256, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=
            True, norm_layer=norm_layer, has_relu=True, has_bias=False),
            ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True,
            norm_layer=norm_layer, has_relu=True, has_bias=False)]
        if is_training:
            heads = [BiSeNetHead(conv_channel, out_planes, 2, True,
                norm_layer), BiSeNetHead(conv_channel, out_planes, 1, True,
                norm_layer), BiSeNetHead(conv_channel * 2, out_planes, 1, 
                False, norm_layer)]
        else:
            heads = [None, None, BiSeNetHead(conv_channel * 2, out_planes, 
                1, False, norm_layer)]
        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1,
            norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)
        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)
        context_blocks = self.backbone(data)
        context_blocks.reverse()
        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context, size=context_blocks[
            0].size()[2:], mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.
            arms, self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=context_blocks[i + 1].size()[2
                :], mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm
        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)
        if self.is_training:
            return pred_out
        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)


def dsn(in_channels, nclass, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3,
        stride=1, padding=1), norm_layer(in_channels), nn.ReLU(), nn.
        Dropout2d(0.1), nn.Conv2d(in_channels, nclass, kernel_size=1,
        stride=1, padding=0, bias=True))


class DFANet(nn.Module):

    def __init__(self, nclass, **kwargs):
        super(DFANet, self).__init__()
        self.backbone = XceptionA()
        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)
        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)
        self.enc2_1_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_2_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_3_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.conv_fusion = ConvBnRelu(32, 32, 1, **kwargs)
        self.fca_1_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_2_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_3_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, nclass, 1)
        self.dsn1 = dsn(192, nclass)
        self.dsn2 = dsn(192, nclass)
        self.__setattr__('exclusive', ['enc2_2', 'enc3_2', 'enc4_2',
            'fca_2', 'enc2_3', 'enc3_3', 'enc3_4', 'fca_3', 'enc2_1_reduce',
            'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion', 'fca_1_reduce',
            'fca_2_reduce', 'fca_3_reduce', 'conv_out'])

    def forward(self, x):
        stage1_conv1 = self.backbone.conv1(x)
        stage1_enc2 = self.backbone.enc2(stage1_conv1)
        stage1_enc3 = self.backbone.enc3(stage1_enc2)
        stage1_enc4 = self.backbone.enc4(stage1_enc3)
        stage1_fca = self.backbone.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode=
            'bilinear', align_corners=True)
        dsn1 = self.dsn1(stage1_out)
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode=
            'bilinear', align_corners=True)
        dsn2 = self.dsn2(stage2_out)
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)
        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2),
            scale_factor=2, mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2),
            scale_factor=4, mode='bilinear', align_corners=True)
        fusion = (stage1_enc2_decoder + stage2_enc2_docoder +
            stage3_enc2_decoder)
        fusion = self.conv_fusion(fusion)
        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca),
            scale_factor=4, mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca),
            scale_factor=8, mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca),
            scale_factor=16, mode='bilinear', align_corners=True)
        fusion = (fusion + stage1_fca_decoder + stage2_fca_decoder +
            stage3_fca_decoder)
        outputs = list()
        out = self.conv_out(fusion)
        outputs.append(out)
        outputs.append(dsn1)
        outputs.append(dsn2)
        return outputs


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class FusionNode(nn.Module):

    def __init__(self, inplane):
        super(FusionNode, self).__init__()
        self.fusion = conv3x3(inplane * 2, inplane)

    def forward(self, x):
        x_h, x_l = x
        size = x_l.size()[2:]
        x_h = F.upsample(x_h, size, mode='bilinear', align_corners=True)
        res = self.fusion(torch.cat([x_h, x_l], dim=1))
        return res


class DFSeg(nn.Module):

    def __init__(self, nclass, type='dfv1'):
        super(DFSeg, self).__init__()
        if type == 'dfv1':
            self.backbone = dfnetv1()
        else:
            self.backbone = dfnetv2()
        self.cc5 = nn.Conv2d(128, 128, 1)
        self.cc4 = nn.Conv2d(256, 128, 1)
        self.cc3 = nn.Conv2d(128, 128, 1)
        self.ppm = PSPModule(512, 128)
        self.fn4 = FusionNode(128)
        self.fn3 = FusionNode(128)
        self.fc = dsn(128, nclass)

    def forward(self, x):
        x3, x4, x5 = self.backbone(x)
        x5 = self.ppm(x5)
        x5 = self.cc5(x5)
        x4 = self.cc4(x4)
        f4 = self.fn4([x5, x4])
        x3 = self.cc3(x3)
        out = self.fn3([f4, x3])
        out = self.fc(out)
        return [out]


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=20, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)
        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.b3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        return classifier


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, classes=20, p=2, q=3, encoderFile=None):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        """
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            None
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=0.001)
        self.conv = CBR(19 + classes, classes, 3, 1)
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * classes),
            DilatedParllelResidualBlockB(2 * classes, classes, add=False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False), BR(classes))
        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2,
            padding=0, output_padding=0, bias=False)

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)
        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)
        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.modules[7](output1_cat)
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))
        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))
        output1_C = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C,
            output2_c], 1)))
        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        out = []
        out.append(classifier)
        return out


class FastSCNN(nn.Module):

    def __init__(self, num_classes, aux=False):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96,
            128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1,
                bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Dropout(
                0.1), nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, dw_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            dw_channels), nn.ReLU(True), nn.Conv2d(dw_channels,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, out_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(_ConvBNReLU(in_channels, in_channels * t,
            1), _DWConv(in_channels * t, in_channels * t, stride), nn.
            Conv2d(in_channels * t, out_channels, 1, bias=False), nn.
            BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64,
        **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
        out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels,
            block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck,
            block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck,
            block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
        scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4,
            mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels,
            num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass,
        norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(nn.Conv2d(low_channels, out_channels,
            3, padding=2, dilation=2, bias=False), norm_layer(out_channels))
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels,
            out_channels, 1, bias=False), norm_layer(out_channels))
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode=
            'bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)
        return x, x_low_cls


class _ICHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(256, 256, 128, nclass, norm_layer)
        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)
        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear',
            align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear',
            align_corners=True)
        outputs.append(up_x8)
        outputs.reverse()
        return outputs


def PSPHead_res50():
    model = PSPHead(Bottleneck, [3, 4, 6, 3])
    return model


class ICNet(nn.Module):

    def __init__(self, nclass):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.Sequential(ConvBnRelu(3, 32, 3, 2, 1),
            ConvBnRelu(32, 32, 3, 2, 1), ConvBnRelu(32, 64, 3, 2, 1))
        self.backbone = PSPHead_res50()
        self.head = _ICHead(nclass)
        self.conv_sub4 = ConvBnRelu(512, 256, 1)
        self.conv_sub2 = ConvBnRelu(512, 256, 1)

    def forward(self, x):
        x_sub1_out = self.conv_sub1(x)
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear',
            align_corners=True)
        x = self.backbone.relu1(self.backbone.bn1(self.backbone.conv1(x_sub2)))
        x = self.backbone.relu2(self.backbone.bn2(self.backbone.conv2(x)))
        x = self.backbone.relu3(self.backbone.bn3(self.backbone.conv3(x)))
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x_sub2_out = self.backbone.layer2(x)
        x_sub4 = F.interpolate(x_sub2_out, scale_factor=0.5, mode=
            'bilinear', align_corners=True)
        x = self.backbone.layer3(x_sub4)
        x = self.backbone.layer4(x)
        x_sub4_out = self.backbone.head(x)
        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)
        res = self.head(x_sub1_out, x_sub2_out, x_sub4_out)
        return res


class MSFNet(nn.Module):

    def __init__(self):
        super(MSFNet, self).__init__()

    def forward(self, x):
        pass


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation * multi_grid, dilation=dilation * multi_grid,
            bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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
        out = self.relu_inplace(out)
        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features,
            out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(features + len(sizes) *
            out_features, out_features, kernel_size=3, padding=1, dilation=
            1, bias=False), BatchNorm2d(out_features), nn.ReLU(), nn.
            Dropout2d(0.1))

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode=
            'bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


affine_par = True


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.head = nn.Sequential(PSPModule(2048, 512), nn.Conv2d(512,
            num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride
            =1, padding=1), BatchNorm2d(512), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = None
        if self.training:
            x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        if self.training:
            return [x, x_dsn]
        else:
            return [x]


class PSPHead(nn.Module):
    """
        Used for ICNet
    """

    def __init__(self, block, layers):
        self.inplanes = 128
        super(PSPHead, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.head = PSPModule(2048, 512)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

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
        relu = self.relu(out)
        return relu, out


class SwiftNetResNet(nn.Module):

    def __init__(self, block, layers, num_features=19, k_up=3, efficient=
        True, use_bn=True, spp_grids=(8, 4, 2, 1), spp_square_grid=False):
        super(SwiftNetResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.nclass = num_features
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features,
            use_bn=self.use_bn, k=k_up)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features,
            use_bn=self.use_bn, k=k_up)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features,
            use_bn=self.use_bn, k=k_up)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.
            layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]
        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size
        level_size = self.spp_size // num_levels
        self.dsn = dsn(256, self.nclass)
        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size
            =bt_size, level_size=level_size, out_size=self.spp_size, grids=
            spp_grids, square_grid=spp_square_grid, bn_momentum=0.01 / 2,
            use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))
        self.random_init = [self.spp, self.upsample]
        self.num_features = num_features
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample,
            efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.
                efficient, use_bn=self.use_bn)]
        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        dsn = None
        if self.training:
            dsn = self.dsn(x)
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        if self.training:
            return features, dsn
        else:
            return features

    def forward_up(self, features):
        features = features[::-1]
        x = features[0]
        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return [x]

    def forward(self, x):
        dsn = None
        if self.training:
            features, dsn = self.forward_down(x)
        else:
            features = self.forward_down(x)
        res = self.forward_up(features)
        if self.training:
            res.append(dsn)
        return res


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear',
    align_corners=True)


class SpatialPyramidPooling(nn.Module):
    """
        SPP module is little different from ppm by inserting middle level feature to save the computation and  memory.
    """

    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128,
        out_size=128, grids=(6, 3, 2, 1), square_grid=False, bn_momentum=
        0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1,
            bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i), _BNReluConv(num_features,
                level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse', _BNReluConv(final_size, out_size, k
            =1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]
        ar = target_size[1] / target_size[0]
        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1
        for i in range(1, num):
            if not self.square_grid:
                grid_size = self.grids[i - 1], max(1, round(ar * self.grids
                    [i - 1]))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)
            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _BNReluConv(nn.Sequential):

    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
        bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=
                bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
            kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):

    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True,
        k=3):
        super(_Upsample, self).__init__()
        None
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1,
            batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k,
            batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
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
        out += residual
        out = self.relu(out)
        return out


class dfnetv1(nn.Module):

    def __init__(self, num_classes=1000):
        super(dfnetv1, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding
            =1, stride=2, bias=False), BatchNorm2d(32), nn.ReLU(inplace=
            True), nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2,
            bias=False), BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage2 = self._make_layer(64, 3, stride=2)
        self.stage3 = self._make_layer(128, 3, stride=2)
        self.stage4 = self._make_layer(256, 3, stride=2)
        self.stage5 = self._make_layer(512, 1, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                BasicBlock.expansion, kernel_size=1, stride=stride, bias=
                False), BatchNorm2d(planes * BasicBlock.expansion))
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x3, x4, x5


class dfnetv2(nn.Module):

    def __init__(self, num_classes=1000):
        super(dfnetv2, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding
            =1, stride=2, bias=False), BatchNorm2d(32), nn.ReLU(inplace=
            True), nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2,
            bias=False), BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage2_1 = self._make_layer(64, 2, stride=2)
        self.stage2_2 = self._make_layer(128, 1, stride=1)
        self.stage3_1 = self._make_layer(128, 10, stride=2)
        self.stage3_2 = self._make_layer(256, 1, stride=1)
        self.stage4_1 = self._make_layer(256, 4, stride=2)
        self.stage4_2 = self._make_layer(512, 2, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                BasicBlock.expansion, kernel_size=1, stride=stride, bias=
                False), BatchNorm2d(planes * BasicBlock.expansion))
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2_1(x)
        x3 = self.stage2_2(x)
        x4 = self.stage3_1(x3)
        x4 = self.stage3_2(x4)
        x5 = self.stage4_1(x4)
        x5 = self.stage4_2(x5)
        return x3, x4, x5


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=
        1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=
        1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps, momentum
            =bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

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
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=
        1e-05, bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size
                =3, stride=2, padding=1, bias=False), norm_layer(stem_width,
                eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                padding=1, bias=False), norm_layer(stem_width, eps=bn_eps,
                momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(
                stem_width, stem_width * 2, kernel_size=3, stride=1,
                padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=
            bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
            inplace, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
            inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
            inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
            inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
        stride=1, bn_eps=1e-05, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps, momentum=
                bn_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer,
            bn_eps, bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=
                norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace
                =inplace))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layers = []
        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(x)
        layers.append(x)
        x = self.layer3(x)
        layers.append(x)
        x = self.layer4(x)
        layers.append(x)
        return layers


class SeparableConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=False)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
            has_bn=True, norm_layer=norm_layer, has_relu=has_relu, has_bias
            =False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_cbr(x)
        return x


class Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_out_channels, has_proj, stride,
        dilation=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.has_proj = has_proj
        if has_proj:
            self.proj = SeparableConvBnRelu(in_channels, mid_out_channels *
                self.expansion, 3, stride, 1, has_relu=False, norm_layer=
                norm_layer)
        self.residual_branch = nn.Sequential(SeparableConvBnRelu(
            in_channels, mid_out_channels, 3, stride, dilation, dilation,
            has_relu=True, norm_layer=norm_layer), SeparableConvBnRelu(
            mid_out_channels, mid_out_channels, 3, 1, 1, has_relu=True,
            norm_layer=norm_layer), SeparableConvBnRelu(mid_out_channels, 
            mid_out_channels * self.expansion, 3, 1, 1, has_relu=False,
            norm_layer=norm_layer))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        if self.has_proj:
            shortcut = self.proj(x)
        residual = self.residual_branch(x)
        output = self.relu(shortcut + residual)
        return output


class Xception(nn.Module):

    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()
        self.in_channels = 8
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1, has_bn=True,
            norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, layers[0],
            channels[0], stride=2)
        self.layer2 = self._make_layer(block, norm_layer, layers[1],
            channels[1], stride=2)
        self.layer3 = self._make_layer(block, norm_layer, layers[2],
            channels[2], stride=2)

    def _make_layer(self, block, norm_layer, blocks, mid_out_channels, stride=1
        ):
        layers = []
        has_proj = True if stride > 1 else False
        layers.append(block(self.in_channels, mid_out_channels, has_proj,
            stride=stride, norm_layer=norm_layer))
        self.in_channels = mid_out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, mid_out_channels,
                has_proj=False, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        return blocks


class BlockA(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
        norm_layer=nn.BatchNorm2d, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride,
                bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4
        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3,
                stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1,
                norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, norm_layer=nn.
        BatchNorm2d):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=
            norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=
                norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(nn.Conv2d(1000, in_channels, 1, bias=
            False), norm_layer(in_channels), nn.ReLU())

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):

    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            norm_layer(8), nn.ReLU())
        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)
        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lxtGH_Fast_Seg(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DFANet(*[], **{'nclass': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FastSCNN(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_003(self):
        self._check(MSFNet(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(ConvBnRelu(*[], **{'in_planes': 4, 'out_planes': 4, 'ksize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ASPPModule(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(A2Block(*[], **{'inplane': 4, 'plane': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(PSPModule(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(AttentionRefinement(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(FeatureFusion(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    def test_011(self):
        self._check(SpatialPath(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(BiSeNetHead(*[], **{'in_planes': 4, 'out_planes': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(BiSeNet(*[], **{'out_planes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_014(self):
        self._check(CBR(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(BR(*[], **{'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(CB(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(C(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(CDilated(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(DownSamplerB(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_020(self):
        self._check(DilatedParllelResidualBlockB(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_021(self):
        self._check(InputProjectionA(*[], **{'samplingTimes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_022(self):
        self._check(ESPNet_Encoder(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_023(self):
        self._check(_ConvBNReLU(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(_DSConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_025(self):
        self._check(_DWConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_026(self):
        self._check(LinearBottleneck(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_027(self):
        self._check(PyramidPooling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_028(self):
        self._check(LearningToDownsample(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_029(self):
        self._check(GlobalFeatureExtractor(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_030(self):
        self._check(FeatureFusionModule(*[], **{'highter_in_channels': 4, 'lower_in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 4, 4])], {})

    def test_031(self):
        self._check(Classifer(*[], **{'dw_channels': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_032(self):
        self._check(CascadeFeatureFusion(*[], **{'low_channels': 4, 'high_channels': 4, 'out_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_033(self):
        self._check(SpatialPyramidPooling(*[], **{'num_maps_in': 4, 'num_levels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_034(self):
        self._check(_BNReluConv(*[], **{'num_maps_in': 4, 'num_maps_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_035(self):
        self._check(_Upsample(*[], **{'num_maps_in': 4, 'skip_maps_in': 4, 'num_maps_out': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_036(self):
        self._check(SeparableConvBnRelu(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_037(self):
        self._check(Block(*[], **{'in_channels': 4, 'mid_out_channels': 4, 'has_proj': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_038(self):
        self._check(BlockA(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_039(self):
        self._check(Enc(*[], **{'in_channels': 4, 'out_channels': 4, 'blocks': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_040(self):
        self._check(FCAttention(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_041(self):
        self._check(XceptionA(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

