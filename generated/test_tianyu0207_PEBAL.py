import sys
_module = sys.modules[__name__]
del sys
config = _module
base_dataset = _module
data_loader = _module
engine = _module
evaluator = _module
lr_policy = _module
trainer = _module
losses = _module
main = _module
mynn = _module
network = _module
resnet = _module
wide_network = _module
wide_resnet_base = _module
wider_resnet = _module
test = _module
conv_2_5d = _module
img_utils = _module
logger = _module
metric = _module
pyt_utils = _module
wandb_upload = _module
valid = _module

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


import torch.utils.data as data


import random


from collections import namedtuple


from typing import Any


from typing import Callable


from typing import Optional


from typing import Tuple


from torch.utils import data


from torch.utils.data import Dataset


import time


import torch.distributed as dist


import collections


import numpy


import torch.nn.functional as F


from torchvision import transforms


import torch.optim


import torch.nn as nn


import logging


from collections import OrderedDict


from functools import partial


from torch.nn.parameter import Parameter


import numbers


import torchvision.transforms as trans


from collections import Counter


import sklearn.metrics as sk


from sklearn.metrics import roc_curve


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import average_precision_score


from sklearn.metrics import auc


import matplotlib.pyplot as plt


import torchvision


from matplotlib.lines import Line2D


from sklearn.preprocessing import minmax_scale as scaler


import warnings


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


class SlidingEval(torch.nn.Module):

    def __init__(self, config, device):
        super(SlidingEval, self).__init__()
        self.config = config
        self.device = device

    def forward(self, img, model, device=None):
        ori_rows, ori_cols, c = img.shape
        num_class = 19
        processed_pred = numpy.zeros((ori_rows, ori_cols, num_class))
        multi_scales = self.config.eval_scale_array
        for s in multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale, (ori_rows, ori_cols), self.config.eval_crop_size, self.config.eval_stride_rate, model, device)
        pred = processed_pred.argmax(2)
        return pred

    def process_image(self, img, crop_size=None):
        p_img = img
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = numpy.concatenate((im_b, im_g, im_r), axis=2)
        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            return p_img, margin
        p_img = p_img.transpose(2, 0, 1)
        return p_img

    def val_func_process(self, input_data, val_func, device=None):
        input_data = numpy.ascontiguousarray(input_data[None, :, :, :], dtype=numpy.float32)
        input_data = torch.FloatTensor(input_data)
        with torch.device(input_data.get_device()):
            val_func.eval()
            val_func
            with torch.no_grad():
                start_time = time.time()
                score = val_func.module(input_data, output_anomaly=False)
                end_time = time.time()
                time_len = end_time - start_time
                score = score.squeeze()[:19]
        return score, time_len

    def scale_process(self, img, ori_shape, crop_size, stride_rate, model, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows
        if isinstance(crop_size, int):
            crop_size = crop_size, crop_size
        class_num = 19
        if long_size <= min(crop_size[0], crop_size[1]):
            input_data, margin = self.process_image(img, crop_size)
            score, _ = self.val_func_process(input_data, model, device)
            score = score[:class_num, margin[0]:score.shape[1] - margin[1], margin[2]:score.shape[2] - margin[3]]
        else:
            stride_0 = int(numpy.ceil(crop_size[0] * stride_rate))
            stride_1 = int(numpy.ceil(crop_size[1] * stride_rate))
            img_pad, margin = self.pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(numpy.ceil((pad_rows - crop_size[0]) / stride_0)) + 1
            c_grid = int(numpy.ceil((pad_cols - crop_size[1]) / stride_1)) + 1
            data_scale = torch.zeros(class_num, pad_rows, pad_cols)
            count_scale = torch.zeros(class_num, pad_rows, pad_cols)
            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]
                    img_sub = img_pad[s_y:e_y, s_x:e_x, :]
                    count_scale[:, s_y:e_y, s_x:e_x] += 1
                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score, _ = self.val_func_process(input_data, model, device)
                    temp_score = temp_score[:class_num, tmargin[0]:temp_score.shape[1] - tmargin[1], tmargin[2]:temp_score.shape[2] - tmargin[3]]
                    data_scale[:, s_y:e_y, s_x:e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:score.shape[1] - margin[1], margin[2]:score.shape[2] - margin[3]]
        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
        return data_output

    def compute_metric(self, results):
        hist = numpy.zeros((19, 19))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1
        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        return mean_IU, mean_pixel_acc

    def pad_image_to_shape(self, img, shape, border_mode, value):
        margin = numpy.zeros(4, numpy.uint32)
        shape = self.get_2dshape(shape)
        pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
        pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0
        margin[0] = pad_height // 2
        margin[1] = pad_height // 2 + pad_height % 2
        margin[2] = pad_width // 2
        margin[3] = pad_width // 2 + pad_width % 2
        img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value)
        return img, margin

    def get_2dshape(self, shape, *, zero=True):
        if not isinstance(shape, collections.Iterable):
            shape = int(shape)
            shape = shape, shape
        else:
            h, w = map(int, shape)
            shape = h, w
        if zero:
            minv = 0
        else:
            minv = 1
        assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
        return shape

    @staticmethod
    def normalize(img, mean, std):
        img = img.astype(numpy.float32) / 255.0
        img = img - mean
        img = img / std
        return img


Norm2d = torch.nn.BatchNorm2d


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling over the input's spatial dimensions
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info('Global Average Pooling Initialized')

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


def bnrelu(channels):
    """
    Single Layer BN and Relui
    """
    return nn.Sequential(mynn.Norm2d(channels), nn.ReLU(inplace=True))


class IdentityResidualBlock(nn.Module):
    """
    Identity Residual Block for WideResnet
    """

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=bnrelu, dropout=None, dist_bn=False):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class WiderResNetA2(nn.Module):
    """
    Wider ResNet with pre-activation (identity mapping) blocks

    This variant uses down-sampling by max-pooling in the first two blocks and
     by strided convolution in the others.

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the six modules of the network.
    norm_act : callable
        Function to create normalization / activation Module.
    classes : int
        If not `0` also include global average pooling and a fully-connected layer
        with `classes` outputs at the end
        of the network.
    dilation : bool
        If `True` apply dilation to the last three modules and change the
        down-sampling factor from 32 to 8.
    """

    def __init__(self, structure, norm_act=bnrelu, classes=0, dilation=False, dist_bn=False):
        super(WiderResNetA2, self).__init__()
        self.dist_bn = dist_bn
        nn.Dropout = nn.Dropout2d
        norm_act = bnrelu
        self.structure = structure
        self.dilation = dilation
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = torch.nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = partial(nn.Dropout, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout, p=0.5)
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop, dist_bn=self.dist_bn)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            return self.classifier(out)
        return out


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()
        if output_stride == 8:
            rates = [(2 * r) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)
        self.features = []
        self.features.append(nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        for r in rates:
            self.features.append(nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False), Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        self.features = torch.nn.ModuleList(self.features)
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DeepWV3Plus(torch.nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7
      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38'):
        super(DeepWV3Plus, self).__init__()
        wide_resnet = WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
        self.gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet
        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)
        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False), Norm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), Norm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
        initialize_weights(self.final)

    def compute_anomaly_score(self, score):
        score = score.squeeze()[:19]
        anomaly_score = -(1.0 * torch.logsumexp(score[:19, :, :] / 1.0, dim=0))
        anomaly_score = anomaly_score.unsqueeze(0)
        anomaly_score = self.gaussian_smoothing(anomaly_score)
        anomaly_score = anomaly_score.squeeze(0)
        return anomaly_score

    def forward(self, inp, output_anomaly=False):
        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0)
        assert len(inp.shape) == 4
        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)
        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        out = Upsample(dec1, x_size[2:])
        if output_anomaly is True:
            anomaly_score = self.compute_anomaly_score(out)
            return anomaly_score
        else:
            return out


class Network(nn.Module):

    def __init__(self, num_classes, wide=False):
        super(Network, self).__init__()
        self.branch1 = DeepWV3Plus(num_classes)

    def forward(self, data, output_anomaly=False):
        return self.branch1(data, output_anomaly=output_anomaly)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
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

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps, momentum=bn_momentum)
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

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0], inplace, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True, stride=1, bn_eps=1e-05, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion, eps=bn_eps, momentum=bn_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace=inplace))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        return blocks


class WiderResNet(nn.Module):
    """
    WideResnet Global Module for Initialization
    """

    def __init__(self, structure, norm_act=bnrelu, classes=0):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and             a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act)))
                in_channels = channels[mod_id][-1]
            if mod_id <= 4:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            out = self.classifier(out)
        return out


def _ntuple(n):

    def parse(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return tuple([x] * n)
    return parse


_pair = _ntuple(2)


class Conv2_5D_disp(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, pixel_size=16):
        super(Conv2_5D_disp, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, disp, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic, extrinsic = camera_params['intrinsic'], camera_params['extrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        disp_col = F.unfold(disp, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        valid_mask = 1 - disp_col.eq(0.0)
        valid_mask *= valid_mask[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        disp_col *= valid_mask
        depth_col = (extrinsic['baseline'] * intrinsic['fx']).view(N, 1, 1) / torch.clamp(disp_col, 0.01, 256)
        valid_mask = valid_mask.view(N, 1, self.kernel_size_prod, out_H * out_W)
        center_depth = depth_col[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N, 1, 1)
        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        mask_1 = torch.abs(depth_col - center_depth).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        mask_1 = (mask_1 + 1 - valid_mask).clamp(min=0.0, max=1.0)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod), (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod), (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod), (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2_5D_depth(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, pixel_size=1, is_graph=False):
        super(Conv2_5D_depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        self.is_graph = is_graph
        if self.is_graph:
            self.weight_0 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
        else:
            self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        center_depth = depth_col[:, self.kernel_size_prod // 2, :]
        center_depth = center_depth.view(N, 1, out_H * out_W)
        grid_range = self.pixel_size * center_depth / intrinsic['fx'].view(N, 1, 1)
        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        mask_1 = torch.abs(depth_col - center_depth).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W)
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod), (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod), (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod), (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeepWV3Plus,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityResidualBlock,
     lambda: ([], {'in_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Network,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_AtrousSpatialPyramidPoolingModule,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_tianyu0207_PEBAL(_paritybench_base):
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

