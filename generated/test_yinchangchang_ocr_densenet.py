import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
densenet = _module
main = _module
resnet = _module
tools = _module
measures = _module
parse = _module
plot = _module
py_op = _module
segmentation = _module
utils = _module
analysis_dataset = _module
map_word_to_index = _module
show_black = _module

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


import numpy as np


from torch.utils.data import Dataset


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import torchvision


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


from torch.utils.data import DataLoader


from sklearn.metrics import roc_auc_score


import torchvision.datasets as dsets


import torchvision.transforms as transforms


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, use_pool):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if use_pool:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), small=0, num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if small and i > 0:
                    use_pool = 0
                else:
                    use_pool = 1
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, use_pool=use_pool)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        return features
        att_feats = features
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        fc_feats = out
        out = self.classifier(out)
        return att_feats, fc_feats, out


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.inplanes = 1024
        self.densenet121 = densenet.densenet121(pretrained=True, small=args.small)
        num_ftrs = self.densenet121.classifier.in_features
        self.classifier_font = nn.Sequential(nn.Conv2d(num_ftrs, out_size, kernel_size=1, bias=False))
        self.train_params = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, phase='train'):
        feats = self.densenet121(x)
        if not args.small:
            feats = F.max_pool2d(feats, kernel_size=2, stride=2)
        out = self.classifier_font(feats)
        out_size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        if phase == 'train':
            out = F.adaptive_max_pool1d(out, output_size=1).view(out.size(0), -1)
            return out
        else:
            out = out.transpose(1, 2).contiguous()
            out = out.view(out_size[0], out_size[2], out_size[3], out_size[1])
            return out, feats


def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, font_output, font_target, weight=None, use_hard_mining=False):
        font_output = self.sigmoid(font_output)
        font_loss = F.binary_cross_entropy(font_output, font_target, weight)
        if use_hard_mining:
            font_output = font_output.view(-1)
            font_target = font_target.view(-1)
            pos_index = font_target > 0.5
            neg_index = font_target == 0
            pos_output = font_output[pos_index]
            pos_target = font_target[pos_index]
            num_hard_pos = max(len(pos_output) / 4, min(5, len(pos_output)))
            if len(pos_output) > 5:
                pos_output, pos_target = hard_mining(pos_output, pos_target, num_hard_pos, largest=False)
            pos_loss = self.classify_loss(pos_output, pos_target) * 0.5
            num_hard_neg = len(pos_output) * 2
            neg_output = font_output[neg_index]
            neg_target = font_target[neg_index]
            neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard_neg, largest=True)
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5
            font_loss += pos_loss + neg_loss
        else:
            pos_loss, neg_loss = font_loss, font_loss
        return [font_loss, pos_loss, neg_loss]

    def _forward(self, font_output, font_target, weight, bbox_output=None, bbox_label=None, seg_output=None, seg_labels=None):
        font_output = self.sigmoid(font_output)
        font_loss = F.binary_cross_entropy(font_output, font_target, weight)
        acc = []
        if bbox_output is not None:
            bbox_output = bbox_output.view((-1, 4))
            bbox_label = bbox_label.view((-1, 4))
            pos_index = bbox_label[:, (-1)] >= 0.5
            pos_index = pos_index.unsqueeze(1).expand(pos_index.size(0), 4)
            neg_index = bbox_label[:, (-1)] <= -0.5
            neg_index = neg_index.unsqueeze(1).expand(neg_index.size(0), 4)
            pos_label = bbox_label[pos_index].view((-1, 4))
            pos_output = bbox_output[pos_index].view((-1, 4))
            lx, ly, ld, lc = pos_label[:, (0)], pos_label[:, (1)], pos_label[:, (2)], pos_label[:, (3)]
            ox, oy, od, oc = pos_output[:, (0)], pos_output[:, (1)], pos_output[:, (2)], pos_output[:, (3)]
            regress_loss = [self.regress_loss(ox, lx), self.regress_loss(oy, ly), self.regress_loss(od, ld)]
            pc = self.sigmoid(oc)
            acc.append((pc >= 0.5).data.cpu().numpy().astype(np.float32).sum())
            acc.append(len(pc))
            classify_loss = self.classify_loss(pc, lc) * 0.5
            neg_label = bbox_label[neg_index].view((-1, 4))
            neg_output = bbox_output[neg_index].view((-1, 4))
            lc = neg_label[:, (3)]
            oc = neg_output[:, (3)]
            pc = self.sigmoid(oc)
            acc.append((pc <= 0.5).data.cpu().numpy().astype(np.float32).sum())
            acc.append(len(pc))
            classify_loss += self.classify_loss(pc, lc + 1) * 0.5
            seg_output = seg_output.view(-1)
            seg_labels = seg_labels.view(-1)
            pos_index = seg_labels > 0.5
            neg_index = seg_labels < 0.5
            seg_loss = 0.5 * self.classify_loss(seg_output[pos_index], seg_labels[pos_index]) + 0.5 * self.classify_loss(seg_output[neg_index], seg_labels[neg_index])
            seg_tpr = (seg_output[pos_index] > 0.5).data.cpu().numpy().astype(np.float32).sum() / len(seg_labels[pos_index])
            seg_tnr = (seg_output[neg_index] < 0.5).data.cpu().numpy().astype(np.float32).sum() / len(seg_labels[neg_index])
        else:
            return font_loss
        if args.model == 'resnet':
            loss = font_loss + classify_loss + seg_loss
        else:
            loss = font_loss + classify_loss + seg_loss
        for reg in regress_loss:
            loss += reg
        return [loss, font_loss, seg_loss, classify_loss] + regress_loss + acc + [seg_tpr, seg_tnr]
        font_num = font_target.sum(0).data.cpu().numpy()
        font_loss = 0
        for di in range(font_num.shape[0]):
            if font_num[di] > 0:
                font_output_i = font_output[:, (di)]
                font_target_i = font_target[:, (di)]
                pos_font_index = font_target_i > 0.5
                font_loss += 0.5 * self.classify_loss(font_output_i[pos_font_index], font_target_i[pos_font_index])
                neg_font_index = font_target_i < 0.5
                if len(font_target_i[neg_font_index]) > 0:
                    font_loss += 0.5 * self.classify_loss(font_output_i[neg_font_index], font_target_i[neg_font_index])
        font_loss = font_loss / (font_num > 0).sum()
        return font_loss


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block=ResidualBlock, layers=[2, 3], num_classes=10, args=None):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0], 2)
        self.layer2 = self.make_layer(block, 64, layers[0], 2)
        self.layer3 = self.make_layer(block, 128, layers[0], 2)
        self.layer4 = self.make_layer(block, 128, layers[0], 2)
        self.layer5 = self.make_layer(block, 128, layers[0], 2)
        self.fc = nn.Linear(128, num_classes)
        self.convt1 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.convt2 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.convt3 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.convt4 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.in_channels = 256
        self.dec1 = self.make_layer(block, 128, layers[0])
        self.in_channels = 256
        self.dec2 = self.make_layer(block, 128, layers[0])
        self.in_channels = 192
        self.dec3 = self.make_layer(block, 128, layers[0])
        self.in_channels = 160
        self.dec4 = nn.Sequential(nn.Conv2d(160, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 1, kernel_size=1, bias=True))
        self.in_channels = 256
        self.bbox = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 4 * len(args.anchors), kernel_size=1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, phase='train'):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        if phase == 'seg':
            out = F.adaptive_max_pool2d(out5, output_size=(1, 1)).view(out.size(0), -1)
            out = self.fc(out)
            out = out.view(out.size(0), -1)
        else:
            out = F.max_pool2d(out5, 2)
            out_size = out.size()
            out = out.view(out_size[0], out_size[1], out_size[2] * out_size[3]).transpose(1, 2).contiguous().view(-1, out_size[1])
            out = self.fc(out)
            out = out.view(out_size[0], out_size[2] * out_size[3], -1).transpose(1, 2).contiguous()
            out = F.adaptive_max_pool1d(out, output_size=1).view(out_size[0], -1)
        if phase not in ['seg', 'pretrain', 'pretrain2']:
            return out
        cat1 = torch.cat([self.convt1(out5), out4], 1)
        dec1 = self.dec1(cat1)
        cat2 = torch.cat([self.convt2(dec1), out3], 1)
        dec2 = self.dec2(cat2)
        cat3 = torch.cat([self.convt3(dec2), out2], 1)
        dec3 = self.dec3(cat3)
        cat4 = torch.cat([self.convt4(dec3), out1], 1)
        seg = self.dec4(cat4)
        seg = seg.view((seg.size(0), seg.size(2), seg.size(3)))
        seg = self.sigmoid(seg)
        bbox = self.bbox(cat2)
        size = bbox.size()
        bbox = bbox.view((size[0], size[1], -1)).transpose(1, 2).contiguous()
        bbox = bbox.view((size[0], size[2], size[3], -1, 4))
        return out, bbox, seg


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4, 'use_pool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yinchangchang_ocr_densenet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

