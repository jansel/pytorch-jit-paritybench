import sys
_module = sys.modules[__name__]
del sys
evaluation = _module
loss = _module
DSOD = _module
RUN = _module
SSD = _module
vgg = _module
multibox = _module
pascal_voc = _module
train = _module
transforms = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


from collections import OrderedDict


import itertools


def log_sum_exp(x, dim, keepdim=False):
    x_max = x.max(dim=dim, keepdim=True)[0]
    if keepdim:
        return (x - x_max).exp().sum(dim=dim, keepdim=True).log() + x_max
    else:
        return (x - x_max).exp().sum(dim=dim).log() + x_max.squeeze(dim)


def _softmax_cross_entropy_with_logits(x, t):
    assert x.size()[:-1] == t.size()
    xt = torch.gather(x, -1, t.long().unsqueeze(-1))
    return log_sum_exp(x, dim=-1, keepdim=False) - xt.squeeze(-1)


class MultiBoxLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def _hard_negative_mining(self, loss, pos, neg, k):
        loss = loss.detach()
        rank = (loss * (-1 * neg.float())).sort(dim=1)[1].sort(dim=1)[1]
        hard_neg = rank < pos.long().sum(dim=1, keepdim=True) * k
        return hard_neg

    def forward(self, xloc, xconf, loc, label, k=3):
        pos = label > 0
        neg = label == 0
        label = label.clamp(min=0)
        pos_idx = pos.unsqueeze(-1).expand_as(xloc)
        loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx]
            .view(-1, 4), size_average=False)
        conf_loss = _softmax_cross_entropy_with_logits(xconf, label)
        hard_neg = self._hard_negative_mining(conf_loss, pos, neg, k)
        conf_loss = conf_loss * (pos + hard_neg).gt(0).float()
        conf_loss = conf_loss.sum()
        N = pos.data.float().sum() + 0.001
        return loc_loss / N, conf_loss / N


def _softmax_focal_loss(x, t, gamma=2):
    assert x.size()[:-1] == t.size()
    logp = torch.gather(F.log_softmax(x), -1, t.long().unsqueeze(-1))
    FL = -(1 - logp.exp()).pow(gamma) * logp
    return FL.sum()


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.count = 0
        self.multiloss = MultiBoxLoss()

    def forward(self, xloc, xconf, loc, label):
        pos = label > 0
        neg = label == 0
        pos_idx = pos.unsqueeze(-1).expand_as(xloc)
        loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx]
            .view(-1, 4), size_average=False)
        pos_idx = pos.unsqueeze(-1).expand_as(xconf)
        pos_conf_loss = _softmax_focal_loss(xconf[pos_idx].view(-1, xconf.
            size(-1)), label[pos])
        neg_idx = neg.unsqueeze(-1).expand_as(xconf)
        neg_conf_loss = _softmax_focal_loss(xconf[neg_idx].view(-1, xconf.
            size(-1)), label[neg])
        conf_loss = self.alpha * pos_conf_loss + (1 - self.alpha
            ) * neg_conf_loss
        self.count += 1
        if self.count % 1000 == 0:
            None
            None
        N = pos.float().sum().clamp(min=0.001)
        return loc_loss / N, conf_loss / N


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(np.ones((1, n_channels,
            1, 1))))
        nn.init.constant(self.scale, scale)

    def forward(self, x):
        x = x * x.pow(2).sum(1, keepdim=True).clamp(min=1e-10).rsqrt()
        return self.scale * x


def bn_relu_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
        bias=False))


def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0
    ):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
        stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU
        (inplace=True))


class DSOD300(nn.Module):
    config = {'name': 'DSOD300-64-192-48-1', 'image_size': 300, 'grids': (
        38, 19, 10, 5, 3, 1), 'aspect_ratios': ((1 / 2.0, 1, 2), (1 / 3.0, 
        1 / 2.0, 1, 2, 3), (1 / 3.0, 1 / 2.0, 1, 2, 3), (1 / 3.0, 1 / 2.0, 
        1, 2, 3), (1 / 2.0, 1, 2), (1 / 2.0, 1, 2)), 'steps': [(s / 300.0) for
        s in [8, 16, 32, 64, 100, 300]], 'sizes': [(s / 300.0) for s in [30,
        60, 111, 162, 213, 264, 315]]}

    def __init__(self, n_classes, growth_rate=48):
        super(DSOD300, self).__init__()
        self.n_classes = n_classes
        depth = [6, 8, 8, 8]
        channels = [(128 + growth_rate * _) for _ in np.cumsum(depth)]
        self.Stem = nn.Sequential(conv_bn_relu(3, 64, 3, stride=2, padding=
            1), conv_bn_relu(64, 64, 3, padding=1), conv_bn_relu(64, 128, 3,
            padding=1), nn.MaxPool2d(2, ceil_mode=True))
        self.Block12 = nn.Sequential(DenseBlock(128, depth[0], growth_rate),
            Transition(channels[0], channels[0], pool=True, ceil_mode=True),
            DenseBlock(channels[0], depth[1], growth_rate), Transition(
            channels[1], channels[1]))
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv2 = bn_relu_conv(channels[1], 256, 1)
        self.Block34 = nn.Sequential(DenseBlock(channels[1], depth[2],
            growth_rate), Transition(channels[2], channels[2]), DenseBlock(
            channels[2], depth[3], growth_rate), Transition(channels[3], 256))
        self.Extra = nn.ModuleList([LHRH(512, 512, ceil_mode=True), LHRH(
            512, 256, ceil_mode=True), LHRH(256, 256, ceil_mode=True), LHRH
            (256, 256)])
        n_channels = [channels[1], 512, 512, 256, 256, 256]
        self.L2Norm = nn.ModuleList()
        self.Loc = nn.ModuleList()
        self.Conf = nn.ModuleList()
        for i, ar in enumerate(self.config['aspect_ratios']):
            n = len(ar) + 1
            self.L2Norm.append(L2Norm(n_channels[i], 20))
            self.Loc.append(nn.Conv2d(n_channels[i], n * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n * (self.n_classes +
                1), 3, padding=1))
        self.apply(self.weights_init)

    def forward(self, x):
        xs = []
        x = self.Stem(x)
        x = self.Block12(x)
        xs.append(x)
        x = self.pool2(x)
        x2 = self.conv2(x)
        x = self.Block34(x)
        x = torch.cat([x2, x], dim=1)
        xs.append(x)
        for m in self.Extra:
            x = m(x)
            xs.append(x)
        return self._prediction(xs)

    def _prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            x = self.L2Norm[i](x)
            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)
            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0),
                -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def init_parameters(self, x):
        pass


class DenseBlock(nn.Module):

    def __init__(self, in_channels, block_depth, growth_rate=48):
        super(DenseBlock, self).__init__()


        class DenseLayer(nn.Module):

            def __init__(self, in_channels, growth_rate, widen=1, dropout=0.0):
                super(DenseLayer, self).__init__()
                self.conv1 = bn_relu_conv(in_channels, growth_rate * widen, 1)
                self.conv2 = bn_relu_conv(growth_rate * widen, growth_rate,
                    3, padding=1)
                self.dropout = dropout

            def forward(self, x):
                out = self.conv1(x)
                out = self.conv2(out)
                if self.dropout > 0:
                    out = F.dropout(out, p=self.dropout, training=self.training
                        )
                return torch.cat([x, out], 1)
        layers = []
        for i in range(block_depth):
            layers.append(DenseLayer(in_channels + i * growth_rate,
                growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):

    def __init__(self, in_channels, out_channels, pool=False, ceil_mode=
        False, dropout=0.0):
        super(Transition, self).__init__()
        self.conv = bn_relu_conv(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2, ceil_mode=ceil_mode
            ) if pool else nn.Sequential()
        self.dropout = dropout

    def forward(self, x):
        out = self.conv(x)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return self.pool(out)


class LHRH(nn.Module):

    def __init__(self, in_channels, out_channels, widen=1, dropout=0.0,
        ceil_mode=False):
        super(LHRH, self).__init__()
        self.conv1_1 = bn_relu_conv(in_channels, int(out_channels / 2 *
            widen), 1)
        self.conv1_2 = bn_relu_conv(int(out_channels / 2 * widen), 
            out_channels // 2, 3, padding=1 * ceil_mode, stride=2)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=ceil_mode)
        self.conv2 = bn_relu_conv(in_channels, out_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x):
        out1 = self.conv1_2(self.conv1_1(x))
        out2 = self.conv2(self.pool2(x))
        if self.dropout > 0:
            out1 = F.dropout(out1, p=self.dropout, training=self.training)
            out2 = F.dropout(out2, p=self.dropout, training=self.training)
        return torch.cat([out1, out2], 1)


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x
            ) * x
        return out


def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
        stride, padding), nn.ReLU(inplace=True))


def deconv_relu(in_channels, out_channels, in_size, out_size):
    if out_size == 2 * in_size:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
            stride=2, padding=1)
    elif out_size == 2 * in_size - 1:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=1)
    elif out_size == 2 * in_size + 1:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=0)
    else:
        raise ValueError('invalid size')
    return nn.Sequential(dconv, nn.ReLU(inplace=True))


class RUN300(nn.Module):
    config = {'name': 'RUN300-VGG16', 'image_size': 300, 'grids': (38, 19, 
        10, 5, 3, 1), 'aspect_ratios': (1 / 3.0, 1 / 2.0, 1, 2, 3), 'steps':
        [(s / 300.0) for s in [8, 16, 32, 64, 100, 300]], 'sizes': [(s / 
        300.0) for s in (30, 60, 111, 162, 213, 264, 315)]}

    def __init__(self, n_classes):
        super(RUN300, self).__init__()
        self.n_classes = n_classes
        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([('extra1_1', nn.Conv2d(1024,
            256, 1)), ('extra1_2', nn.Conv2d(256, 512, 3, padding=1, stride
            =2)), ('extra2_1', nn.Conv2d(512, 128, 1)), ('extra2_2', nn.
            Conv2d(128, 256, 3, padding=1, stride=2)), ('extra3_1', nn.
            Conv2d(256, 128, 1)), ('extra3_2', nn.Conv2d(128, 256, 3)), (
            'extra4_1', nn.Conv2d(256, 128, 1)), ('extra4_2', nn.Conv2d(128,
            256, 3))]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2',
            'extra3_2', 'extra4_2']
        n_channels = [512, 1024, 512, 256, 256, 256]
        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.l2norm_layers = ['conv4_3']
        self.ResBlocks = nn.ModuleList()
        for i in range(len(n_channels) - 1):
            self.ResBlocks.append(ThreeWay(n_channels[i], n_channels[i + 1],
                self.config['grids'][i], self.config['grids'][i + 1],
                out_channels=256))
        self.ResBlocks.append(TwoWay(n_channels[-1], out_channels=256))
        n_boxes = len(self.config['aspect_ratios']) + 1
        self.Loc = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(inplace=
            True), nn.Conv2d(256, n_boxes * 4, 3, padding=1))
        self.Conf = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(inplace=
            True), nn.Conv2d(256, n_boxes * (self.n_classes + 1), 3, padding=1)
            )

    def forward(self, x):
        xs = []
        for name, m in itertools.chain(self.Base._modules.items(), self.
            Extra._modules.items()):
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)
            if name in self.pred_layers:
                if name in self.l2norm_layers:
                    i = self.l2norm_layers.index(name)
                    xs.append(self.L2Norm[i](x))
                else:
                    xs.append(x)
        return self._prediction(self.multiway(xs))

    def multiway(self, xs):
        ys = []
        for i in range(len(xs)):
            block = self.ResBlocks[i]
            if isinstance(block, ThreeWay):
                y = block(xs[i], xs[i + 1])
                ys.append(y)
            elif isinstance(block, TwoWay):
                y = block(xs[i])
                ys.append(y)
        return ys

    def _prediction(self, ys):
        locs = []
        confs = []
        for y in ys:
            loc = self.Loc(y)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)
            conf = self.Conf(y)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0),
                -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    def init_parameters(self, backbone=None):

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.apply(weights_init)
        if backbone is not None and os.path.isfile(backbone):
            self.Base.load_pretrained(backbone)


class TwoWay(nn.Module):

    def __init__(self, in_channels, out_channels=256, bypass=False):
        super().__init__()
        if bypass and in_channels == out_channels:
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = conv_relu(in_channels, out_channels, 1)
        self.branch2 = nn.Sequential(conv_relu(in_channels, out_channels //
            2, 1), conv_relu(out_channels // 2, out_channels // 2, 3,
            padding=1), conv_relu(out_channels // 2, out_channels, 1))

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x /= x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x
            ) * x
        return out


class SSD300(nn.Module):
    config = {'name': 'SSD300-VGG16', 'image_size': 300, 'grids': (38, 19, 
        10, 5, 3, 1), 'aspect_ratios': ((1 / 2.0, 1, 2), (1 / 3.0, 1 / 2.0,
        1, 2, 3), (1 / 3.0, 1 / 2.0, 1, 2, 3), (1 / 3.0, 1 / 2.0, 1, 2, 3),
        (1 / 2.0, 1, 2), (1 / 2.0, 1, 2)), 'steps': [(s / 300.0) for s in [
        8, 16, 32, 64, 100, 300]], 'sizes': [(s / 300.0) for s in [30, 60, 
        111, 162, 213, 264, 315]], 'prior_variance': [0.1, 0.1, 0.2, 0.2]}

    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([('extra1_1', nn.Conv2d(1024,
            256, 1)), ('extra1_2', nn.Conv2d(256, 512, 3, padding=1, stride
            =2)), ('extra2_1', nn.Conv2d(512, 128, 1)), ('extra2_2', nn.
            Conv2d(128, 256, 3, padding=1, stride=2)), ('extra3_1', nn.
            Conv2d(256, 128, 1)), ('extra3_2', nn.Conv2d(128, 256, 3)), (
            'extra4_1', nn.Conv2d(256, 128, 1)), ('extra4_2', nn.Conv2d(128,
            256, 3))]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2',
            'extra3_2', 'extra4_2']
        n_channels = [512, 1024, 512, 256, 256, 256]
        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.norm_layers = ['conv4_3']
        self.Loc = nn.ModuleList([])
        self.Conf = nn.ModuleList([])
        for i, ar in enumerate(self.config['aspect_ratios']):
            n = len(ar) + 1
            self.Loc.append(nn.Conv2d(n_channels[i], n * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n * (self.n_classes +
                1), 3, padding=1))

    def forward(self, x):
        xs = []
        for name, m in itertools.chain(self.Base._modules.items(), self.
            Extra._modules.items()):
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)
            if name in self.pred_layers:
                if name in self.norm_layers:
                    i = self.norm_layers.index(name)
                    xs.append(self.L2Norm[i](x))
                else:
                    xs.append(x)
        return self._prediction(xs)

    def _prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)
            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0),
                -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    def init_parameters(self, backbone=None):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
        self.apply(weights_init)
        if backbone is not None and os.path.isfile(backbone):
            self.Base.load_pretrained(backbone)
            None
        else:
            None


class VGG16(nn.Module):
    """
    input image: BGR format, range [0, 255], then subtract mean
    """

    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, 1)

    def load_pretrained(self, path):
        weights = torch.load(path)
        lookup = {'conv1_1': '0', 'conv1_2': '2', 'conv2_1': '5', 'conv2_2':
            '7', 'conv3_1': '10', 'conv3_2': '12', 'conv3_3': '14',
            'conv4_1': '17', 'conv4_2': '19', 'conv4_3': '21', 'conv5_1':
            '24', 'conv5_2': '26', 'conv5_3': '28', 'conv6': '31', 'conv7':
            '33'}
        model_dict = self.state_dict()
        pretrained_dict = {}
        for name, ind in lookup.items():
            for ext in ['.weight', '.bias']:
                pretrained_dict[name + ext] = weights[ind + ext]
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_uoip_SSD_variants(_paritybench_base):
    pass
    def test_000(self):
        self._check(DenseBlock(*[], **{'in_channels': 4, 'block_depth': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(L2Norm(*[], **{'n_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Transition(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(TwoWay(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

