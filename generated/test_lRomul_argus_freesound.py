import sys
_module = sys.modules[__name__]
del sys
after_train_folds = _module
blend_kernel_template = _module
blend_predict = _module
build_kernel = _module
kernel_template = _module
make_folds = _module
predict_folds = _module
random_search = _module
src = _module
argus_models = _module
audio = _module
config = _module
datasets = _module
losses = _module
lr_scheduler = _module
metrics = _module
mixers = _module
models = _module
aux_skip_attention = _module
feature_extractor = _module
resnet = _module
rnn_aux_skip_attention = _module
senet = _module
simple_attention = _module
simple_kaggle = _module
skip_attention = _module
predictor = _module
random_resized_crop = _module
stacking = _module
argus_models = _module
models = _module
transforms = _module
tiles = _module
utils = _module
stacking_kernel_template = _module
stacking_predict = _module
stacking_random_search = _module
stacking_val_predict = _module
train_folds = _module
train_stacking = _module

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


import torch


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import math


from torch.utils import model_zoo


def lq_loss(y_pred, y_true, q):
    eps = 1e-07
    loss = y_pred * y_true
    loss = (1 - (loss + eps) ** q) / q
    return loss.mean()


class LqLoss(nn.Module):

    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return lq_loss(output, target, self.q)


def l_soft(y_pred, y_true, beta):
    eps = 1e-07
    y_pred = torch.clamp(y_pred, eps, 1.0)
    with torch.no_grad():
        y_true_update = beta * y_true + (1 - beta) * y_pred
    loss = F.binary_cross_entropy(y_pred, y_true_update)
    return loss


class LSoftLoss(nn.Module):

    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return l_soft(output, target, self.beta)


class NoisyCuratedLoss(nn.Module):

    def __init__(self, noisy_loss, curated_loss, noisy_weight=0.5,
        curated_weight=0.5):
        super().__init__()
        self.noisy_loss = noisy_loss
        self.curated_loss = curated_loss
        self.noisy_weight = noisy_weight
        self.curated_weight = curated_weight

    def forward(self, output, target, noisy):
        batch_size = target.shape[0]
        noisy_indexes = noisy.nonzero().squeeze(1)
        curated_indexes = (noisy == 0).nonzero().squeeze(1)
        noisy_len = noisy_indexes.shape[0]
        if noisy_len > 0:
            noisy_target = target[noisy_indexes]
            noisy_output = output[noisy_indexes]
            noisy_loss = self.noisy_loss(noisy_output, noisy_target)
            noisy_loss = noisy_loss * (noisy_len / batch_size)
        else:
            noisy_loss = 0
        curated_len = curated_indexes.shape[0]
        if curated_len > 0:
            curated_target = target[curated_indexes]
            curated_output = output[curated_indexes]
            curated_loss = self.curated_loss(curated_output, curated_target)
            curated_loss = curated_loss * (curated_len / batch_size)
        else:
            curated_loss = 0
        loss = noisy_loss * self.noisy_weight
        loss += curated_loss * self.curated_weight
        return loss


class OnlyNoisyLqLoss(nn.Module):

    def __init__(self, q=0.5, noisy_weight=0.5, curated_weight=0.5):
        super().__init__()
        lq = LqLoss(q=q)
        bce = nn.BCEWithLogitsLoss()
        self.loss = NoisyCuratedLoss(lq, bce, noisy_weight, curated_weight)

    def forward(self, output, target, noisy):
        return self.loss(output, target, noisy)


class OnlyNoisyLSoftLoss(nn.Module):

    def __init__(self, beta, noisy_weight=0.5, curated_weight=0.5):
        super().__init__()
        soft = LSoftLoss(beta)
        bce = nn.BCEWithLogitsLoss()
        self.loss = NoisyCuratedLoss(soft, bce, noisy_weight, curated_weight)

    def forward(self, output, target, noisy):
        return self.loss(output, target, noisy)


class BCEMaxOutlierLoss(nn.Module):

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target, noisy):
        loss = F.binary_cross_entropy_with_logits(output, target, reduction
            ='none')
        loss = loss.mean(dim=1)
        with torch.no_grad():
            outlier_mask = loss > self.alpha * loss.max()
            outlier_mask = outlier_mask * noisy
            outlier_idx = (outlier_mask == 0).nonzero().squeeze(1)
        loss = loss[outlier_idx].mean()
        return loss


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SkipBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AuxBlock(nn.Module):

    def __init__(self, last_fc, num_classes, base_size, dropout):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size * 
            8, base_size * last_fc), nn.PReLU(), nn.BatchNorm1d(base_size *
            last_fc), nn.Dropout(dropout / 2), nn.Linear(base_size *
            last_fc, num_classes))

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AuxSkipAttention(nn.Module):

    def __init__(self, num_classes, base_size=64, dropout=0.2, ratio=16,
        kernel_size=7, last_filters=8, last_fc=2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=base_size)
        self.skip1 = SkipBlock(in_channels=base_size, out_channels=
            base_size * 8, scale_factor=8)
        self.conv2 = ConvBlock(in_channels=base_size, out_channels=
            base_size * 2)
        self.skip2 = SkipBlock(in_channels=base_size * 2, out_channels=
            base_size * 8, scale_factor=4)
        self.conv3 = ConvBlock(in_channels=base_size * 2, out_channels=
            base_size * 4)
        self.skip3 = SkipBlock(in_channels=base_size * 4, out_channels=
            base_size * 8, scale_factor=2)
        self.conv4 = ConvBlock(in_channels=base_size * 4, out_channels=
            base_size * 8)
        self.attention = ConvolutionalBlockAttentionModule(base_size * 8 * 
            4, ratio=ratio, kernel_size=kernel_size)
        self.merge = SkipBlock(base_size * 8 * 4, base_size * last_filters, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size *
            last_filters, base_size * last_fc), nn.PReLU(), nn.BatchNorm1d(
            base_size * last_fc), nn.Dropout(dropout / 2), nn.Linear(
            base_size * last_fc, num_classes))
        self.aux1 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux2 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux3 = AuxBlock(last_fc, num_classes, base_size, dropout)

    def forward(self, x):
        x = self.conv1(x)
        skip1 = self.skip1(x)
        aux1 = self.aux1(skip1)
        x = self.conv2(x)
        skip2 = self.skip2(x)
        aux2 = self.aux2(skip2)
        x = self.conv3(x)
        skip3 = self.skip3(x)
        aux3 = self.aux3(skip3)
        x = self.conv4(x)
        x = torch.cat([x, skip1, skip2, skip3], dim=1)
        x = self.attention(x)
        x = self.merge(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, aux3, aux2, aux1


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False,
            dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeatureExtractor(nn.Module):

    def __init__(self, num_classes, input_channels=3, base_size=32, dropout
        =0.25):
        super(FeatureExtractor, self).__init__()
        self.input_channels = input_channels
        self.base_size = base_size
        s = base_size
        self.dropout = dropout
        self.input_conv = BasicConv2d(input_channels, s, 1)
        self.conv_1 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_2 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_3 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.conv_4 = BasicConv2d(s * 2, s * 2, 3, padding=1)
        self.conv_5 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.conv_6 = BasicConv2d(s * 4, s * 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(s * 4, num_classes)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        x = self.conv_5(x)
        x = self.pool(x)
        x = self.conv_6(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, norm_layer=norm_layer))
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
        x = self.fc(x)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SkipBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AuxBlock(nn.Module):

    def __init__(self, last_fc, num_classes, base_size, dropout):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size * 
            8, base_size * last_fc), nn.PReLU(), nn.BatchNorm1d(base_size *
            last_fc), nn.Dropout(dropout / 2), nn.Linear(base_size *
            last_fc, num_classes))

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, in_channels, hidden, out_channels):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_channels, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out_channels)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class RnnAuxSkipAttention(nn.Module):

    def __init__(self, num_classes, base_size=64, dropout=0.2, ratio=16,
        kernel_size=7, last_filters=8, last_fc=2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=base_size)
        self.skip1 = SkipBlock(in_channels=base_size, out_channels=
            base_size * 8, scale_factor=8)
        self.conv2 = ConvBlock(in_channels=base_size, out_channels=
            base_size * 2)
        self.skip2 = SkipBlock(in_channels=base_size * 2, out_channels=
            base_size * 8, scale_factor=4)
        self.conv3 = ConvBlock(in_channels=base_size * 2, out_channels=
            base_size * 4)
        self.skip3 = SkipBlock(in_channels=base_size * 4, out_channels=
            base_size * 8, scale_factor=2)
        self.conv4 = ConvBlock(in_channels=base_size * 4, out_channels=
            base_size * 8)
        self.attention = ConvolutionalBlockAttentionModule(base_size * 8 * 
            4, ratio=ratio, kernel_size=kernel_size)
        self.merge = SkipBlock(base_size * 8 * 4, base_size * last_filters, 1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size *
            last_filters, base_size * last_fc), nn.PReLU(), nn.BatchNorm1d(
            base_size * last_fc), nn.Dropout(dropout / 2), nn.Linear(
            base_size * last_fc, num_classes))
        self.aux1 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux2 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux3 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.rnn = BidirectionalLSTM(base_size * last_filters, base_size *
            last_filters, base_size * last_filters)

    def forward(self, x):
        x = self.conv1(x)
        skip1 = self.skip1(x)
        aux1 = self.aux1(skip1)
        x = self.conv2(x)
        skip2 = self.skip2(x)
        aux2 = self.aux2(skip2)
        x = self.conv3(x)
        skip3 = self.skip3(x)
        aux3 = self.aux3(skip3)
        x = self.conv4(x)
        x = torch.cat([x, skip1, skip2, skip3], dim=1)
        x = self.attention(x)
        x = self.merge(x)
        x = torch.mean(x, dim=2, keepdim=True)
        b, c, h, w = x.size()
        assert h == 1, f'the height of conv must be 1, got {h}'
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, aux3, aux2, aux1


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SimpleAttention(nn.Module):

    def __init__(self, num_classes, base_size=64, dropout=0.2, ratio=16,
        kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(in_channels=3, out_channels=
            base_size), ConvBlock(in_channels=base_size, out_channels=
            base_size * 2), ConvBlock(in_channels=base_size * 2,
            out_channels=base_size * 4), ConvBlock(in_channels=base_size * 
            4, out_channels=base_size * 8))
        self.attention = ConvolutionalBlockAttentionModule(base_size * 8,
            ratio=ratio, kernel_size=kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size * 
            8, base_size * 2), nn.PReLU(), nn.BatchNorm1d(base_size * 2),
            nn.Dropout(dropout / 2), nn.Linear(base_size * 2, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
            1, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,
            1, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SimpleKaggle(nn.Module):

    def __init__(self, num_classes, base_size=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(in_channels=3, out_channels=
            base_size), ConvBlock(in_channels=base_size, out_channels=
            base_size * 2), ConvBlock(in_channels=base_size * 2,
            out_channels=base_size * 4), ConvBlock(in_channels=base_size * 
            4, out_channels=base_size * 8))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size * 
            8, base_size * 2), nn.PReLU(), nn.BatchNorm1d(base_size * 2),
            nn.Dropout(dropout / 2), nn.Linear(base_size * 2, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3,
            1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SkipBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SkipAttention(nn.Module):

    def __init__(self, num_classes, base_size=64, dropout=0.2, ratio=16,
        kernel_size=7, last_filters=8, last_fc=2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=base_size)
        self.skip1 = SkipBlock(in_channels=base_size, out_channels=
            base_size * 8, scale_factor=8)
        self.conv2 = ConvBlock(in_channels=base_size, out_channels=
            base_size * 2)
        self.skip2 = SkipBlock(in_channels=base_size * 2, out_channels=
            base_size * 8, scale_factor=4)
        self.conv3 = ConvBlock(in_channels=base_size * 2, out_channels=
            base_size * 4)
        self.skip3 = SkipBlock(in_channels=base_size * 4, out_channels=
            base_size * 8, scale_factor=2)
        self.conv4 = ConvBlock(in_channels=base_size * 4, out_channels=
            base_size * 8)
        self.attention = ConvolutionalBlockAttentionModule(base_size * 8 * 
            4, ratio=ratio, kernel_size=kernel_size)
        self.merge = SkipBlock(base_size * 8 * 4, base_size * last_filters, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_size *
            last_filters, base_size * last_fc), nn.PReLU(), nn.BatchNorm1d(
            base_size * last_fc), nn.Dropout(dropout / 2), nn.Linear(
            base_size * last_fc, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        skip1 = self.skip1(x)
        x = self.conv2(x)
        skip2 = self.skip2(x)
        x = self.conv3(x)
        skip3 = self.skip3(x)
        x = self.conv4(x)
        x = torch.cat([x, skip1, skip2, skip3], dim=1)
        x = self.attention(x)
        x = self.merge(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SEScale(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        channel = in_channels
        self.fc1 = nn.Linear(channel, reduction)
        self.fc2 = nn.Linear(reduction, channel)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class FCNet(nn.Module):

    def __init__(self, in_channels, num_classes, base_size=64,
        reduction_scale=8, p_dropout=0.2):
        super().__init__()
        self.p_dropout = p_dropout
        self.scale = SEScale(in_channels, in_channels // reduction_scale)
        self.linear1 = nn.Linear(in_channels, base_size * 2)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(base_size * 2, base_size)
        self.relu2 = nn.PReLU()
        self.fc = nn.Linear(base_size, num_classes)

    def forward(self, x):
        x = self.scale(x) * x
        x = self.linear1(x)
        x = self.relu1(x)
        if self.p_dropout is not None:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.linear2(x)
        x = self.relu2(x)
        if self.p_dropout is not None:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lRomul_argus_freesound(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LqLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(LSoftLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(NoisyCuratedLoss(*[], **{'noisy_loss': MSELoss(), 'curated_loss': MSELoss()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(OnlyNoisyLqLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(OnlyNoisyLSoftLoss(*[], **{'beta': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(BCEMaxOutlierLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ChannelAttention(*[], **{'in_planes': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_007(self):
        self._check(SpatialAttention(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ConvolutionalBlockAttentionModule(*[], **{'in_planes': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_009(self):
        self._check(ConvBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(SkipBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(AuxSkipAttention(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_012(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(FeatureExtractor(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_014(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(BidirectionalLSTM(*[], **{'in_channels': 4, 'hidden': 4, 'out_channels': 4}), [torch.rand([4, 4, 4])], {})

    def test_016(self):
        self._check(RnnAuxSkipAttention(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_017(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(SimpleAttention(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_019(self):
        self._check(SimpleKaggle(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_020(self):
        self._check(SkipAttention(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_021(self):
        self._check(SEScale(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

