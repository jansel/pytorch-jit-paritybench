import sys
_module = sys.modules[__name__]
del sys
scene_generation = _module
args = _module
bilinear = _module
data = _module
coco = _module
coco_panoptic = _module
utils = _module
discriminators = _module
generators = _module
graph = _module
layers = _module
layout = _module
losses = _module
metrics = _module
model = _module
trainer = _module
utils = _module
vis = _module
create_attributes_file = _module
encode_features = _module
model = _module
inception_score = _module
sample_images = _module
train_accuracy_net = _module
train = _module

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


import torch


import torch.nn.functional as F


import math


import random


from collections import defaultdict


import numpy as np


import torchvision.transforms as T


from torch.utils.data import Dataset


import torch.nn as nn


import functools


from torch.nn.functional import interpolate


from torch import nn


from torchvision import models


from torch.autograd import Variable


import torchvision


from torch.utils.data import DataLoader


from sklearn.cluster import KMeans


from sklearn.manifold import TSNE


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


from scipy.stats import entropy


from torch.nn import functional as F


from torchvision.models.inception import inception_v3


from random import randint


import time


import torch.optim as optim


from torch.optim import lr_scheduler


class GlobalAvgPool(nn.Module):

    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(dim=2)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return 'Flatten()'


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


def _get_padding(K, mode):
    """ Helper method to compute padding size """
    if mode == 'valid':
        return 0
    elif mode == 'same':
        assert K % 2 == 1, 'Invalid kernel size %d for "same" padding' % K
        return (K - 1) // 2


def _init_conv(layer, method):
    if not isinstance(layer, nn.Conv2d):
        return
    if method == 'default':
        return
    elif method == 'kaiming-normal':
        nn.init.kaiming_normal(layer.weight)
    elif method == 'kaiming-uniform':
        nn.init.kaiming_uniform(layer.weight)


def get_activation(name):
    kwargs = {}
    if name.lower().startswith('leakyrelu'):
        if '-' in name:
            slope = float(name.split('-')[1])
            kwargs = {'negative_slope': slope}
    name = 'leakyrelu'
    activations = {'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU}
    if name.lower() not in activations:
        raise ValueError('Invalid activation "%s"' % name)
    return activations[name.lower()](**kwargs)


def get_normalization_2d(channels, normalization):
    if normalization == 'instance':
        return nn.InstanceNorm2d(channels)
    elif normalization == 'batch':
        return nn.BatchNorm2d(channels)
    elif normalization == 'none':
        return None
    else:
        raise ValueError('Unrecognized normalization type "%s"' % normalization)


class ResidualBlock(nn.Module):

    def __init__(self, channels, normalization='batch', activation='relu', padding='same', kernel_size=3, init='default'):
        super(ResidualBlock, self).__init__()
        K = kernel_size
        P = _get_padding(K, padding)
        C = channels
        self.padding = P
        layers = [get_normalization_2d(C, normalization), get_activation(activation), nn.Conv2d(C, C, kernel_size=K, padding=P), get_normalization_2d(C, normalization), get_activation(activation), nn.Conv2d(C, C, kernel_size=K, padding=P)]
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            _init_conv(layer, method=init)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        P = self.padding
        shortcut = x
        if P == 0:
            shortcut = x[:, :, P:-P, P:-P]
        y = self.net(x)
        return shortcut + self.net(x)


def build_cnn(arch, normalization='batch', activation='relu', padding='same', pooling='max', init='default'):
    """
    Build a CNN from an architecture string, which is a list of layer
    specification strings. The overall architecture can be given as a list or as
    a comma-separated string.

    All convolutions *except for the first* are preceeded by normalization and
    nonlinearity.

    All other layers support the following:
    - IX: Indicates that the number of input channels to the network is X.
          Can only be used at the first layer; if not present then we assume
          3 input channels.
    - CK-X: KxK convolution with X output channels
    - CK-X-S: KxK convolution with X output channels and stride S
    - R: Residual block keeping the same number of channels
    - UX: Nearest-neighbor upsampling with factor X
    - PX: Spatial pooling with factor X
    - FC-X-Y: Flatten followed by fully-connected layer

    Returns a tuple of:
    - cnn: An nn.Sequential
    - channels: Number of output channels
    """
    if isinstance(arch, str):
        arch = arch.split(',')
    cur_C = 3
    if len(arch) > 0 and arch[0][0] == 'I':
        cur_C = int(arch[0][1:])
        arch = arch[1:]
    first_conv = True
    flat = False
    layers = []
    for i, s in enumerate(arch):
        if s[0] == 'C':
            if not first_conv:
                layers.append(get_normalization_2d(cur_C, normalization))
                layers.append(get_activation(activation))
            first_conv = False
            vals = [int(i) for i in s[1:].split('-')]
            if len(vals) == 2:
                K, next_C = vals
                stride = 1
            elif len(vals) == 3:
                K, next_C, stride = vals
            P = _get_padding(K, padding)
            conv = nn.Conv2d(cur_C, next_C, kernel_size=K, padding=P, stride=stride)
            layers.append(conv)
            _init_conv(layers[-1], init)
            cur_C = next_C
        elif s[0] == 'R':
            norm = 'none' if first_conv else normalization
            res = ResidualBlock(cur_C, normalization=norm, activation=activation, padding=padding, init=init)
            layers.append(res)
            first_conv = False
        elif s[0] == 'U':
            factor = int(s[1:])
            layers.append(Interpolate(scale_factor=factor, mode='nearest'))
        elif s[0] == 'P':
            factor = int(s[1:])
            if pooling == 'max':
                pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
            elif pooling == 'avg':
                pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
            layers.append(pool)
        elif s[:2] == 'FC':
            _, Din, Dout = s.split('-')
            Din, Dout = int(Din), int(Dout)
            if not flat:
                layers.append(Flatten())
            flat = True
            layers.append(nn.Linear(Din, Dout))
            if i + 1 < len(arch):
                layers.append(get_activation(activation))
            cur_C = Dout
        else:
            raise ValueError('Invalid layer "%s"' % s)
    layers = [layer for layer in layers if layer is not None]
    return nn.Sequential(*layers), cur_C


class AcDiscriminator(nn.Module):

    def __init__(self, vocab, arch, normalization='none', activation='relu', padding='same', pooling='avg'):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab
        cnn_kwargs = {'arch': arch, 'normalization': normalization, 'activation': activation, 'pooling': pooling, 'padding': padding}
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        num_objects = len(vocab['object_to_idx'])
        self.real_classifier = nn.Linear(1024, 1)
        self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, (None)]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


def bilinear_sample(feats, X, Y):
    """
    Perform bilinear sampling on the features in feats using the sampling grid
    given by X and Y.

    Inputs:
    - feats: Tensor holding input feature map, of shape (N, C, H, W)
    - X, Y: Tensors holding x and y coordinates of the sampling
      grids; both have shape shape (N, HH, WW) and have elements in the range [0, 1].
    Returns:
    - out: Tensor of shape (B, C, HH, WW) where out[i] is computed
      by sampling from feats[idx[i]] using the sampling grid (X[i], Y[i]).
    """
    N, C, H, W = feats.size()
    assert X.size() == Y.size()
    assert X.size(0) == N
    _, HH, WW = X.size()
    X = X.mul(W)
    Y = Y.mul(H)
    x0 = X.floor().clamp(min=0, max=W - 1)
    x1 = (x0 + 1).clamp(min=0, max=W - 1)
    y0 = Y.floor().clamp(min=0, max=H - 1)
    y1 = (y0 + 1).clamp(min=0, max=H - 1)
    y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    feats_flat = feats.view(N, C, H * W)
    v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
    v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
    v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
    v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)
    w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)
    start_w = torch.linspace(1, 0, steps=steps)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps)
    end_w = end_w.view(w_size).expand(out_size)
    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)
    out = start_w * start + end_w * end
    return out


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None:
        WW = HH
    if backend == 'cudnn':
        bbox = 2 * bbox - 1
    x0, y0 = bbox[:, (0)], bbox[:, (1)]
    x1, y1 = bbox[:, (2)], bbox[:, (3)]
    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        return F.grid_sample(feats, grid)


def _invperm(p):
    N = p.size(0)
    eye = torch.arange(0, N).type_as(p)
    pp = (eye[:, (None)] == p).nonzero()[:, (1)]
    return pp


def crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW=None):
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None:
        WW = HH
    dtype = feats.data.type()
    feats_flat, bbox_flat, all_idx = [], [], []
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        feats_flat.append(cur_feats)
        bbox_flat.append(cur_bbox)
        all_idx.append(idx)
    feats_flat = torch.cat(feats_flat, dim=0)
    bbox_flat = torch.cat(bbox_flat, dim=0)
    crops = crop_bbox(feats_flat, bbox_flat, HH, WW, backend='cudnn')
    all_idx = torch.cat(all_idx, dim=0)
    eye = torch.arange(0, B).type_as(all_idx)
    if (all_idx == eye).all():
        return crops
    return crops[_invperm(all_idx)]


def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None, backend='cudnn'):
    """
    Inputs:
    - feats: FloatTensor of shape (N, C, H, W)
    - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
      each element is in the range [0, N) and bbox_to_feats[b] = i means that
      bbox[b] will be cropped from feats[i].
    - HH, WW: Size of the output crops

    Returns:
    - crops: FloatTensor of shape (B, C, HH, WW) where crops[i] uses bbox[i] to
      crop from feats[bbox_to_feats[i]].
    """
    if backend == 'cudnn':
        return crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW)
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None:
        WW = HH
    dtype, device = feats.dtype, feats.device
    crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
        crops[idx] = cur_crops
    return crops


class AcCropDiscriminator(nn.Module):

    def __init__(self, vocab, arch, normalization='none', activation='relu', object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.discriminator = AcDiscriminator(vocab, arch, normalization, activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes, obj_to_img):
        crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, ac_loss = self.discriminator(crops, objs)
        return real_scores, ac_loss, crops


class NLayerMaskDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_objects=None):
        super(NLayerMaskDiscriminator, self).__init__()
        self.n_layers = n_layers
        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        nf_prev += num_objects
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


class MultiscaleMaskDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, num_objects=None):
        super(MultiscaleMaskDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        for i in range(num_D):
            netD = NLayerMaskDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_objects)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, cond):
        result = [input]
        for i in range(len(model) - 2):
            result.append(model[i](result[-1]))
        a, b, c, d = result[-1].shape
        cond = cond.view(a, -1, 1, 1).expand(-1, -1, c, d)
        concat = torch.cat([result[-1], cond], dim=1)
        result.append(model[len(model) - 2](concat))
        result.append(model[len(model) - 1](result[-1]))
        return result[1:]

    def forward(self, input, cond):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled, cond))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        result = [input]
        for i in range(len(model)):
            result.append(model[i](result[-1]))
        return result[1:]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class AppearanceEncoder(nn.Module):

    def __init__(self, vocab, arch, normalization='none', activation='relu', padding='same', vecs_size=1024, pooling='avg'):
        super(AppearanceEncoder, self).__init__()
        self.vocab = vocab
        cnn_kwargs = {'arch': arch, 'normalization': normalization, 'activation': activation, 'pooling': pooling, 'padding': padding}
        cnn, channels = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(channels, vecs_size))

    def forward(self, crops):
        return self.cnn(crops)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


def build_mlp(dim_list, activation='relu', batch_norm='none', dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = i == len(dim_list) - 2
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, attributes_dim=0, output_dim=None, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim + 2 * attributes_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)
        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim
        s_idx = edges[:, (0)].contiguous()
        o_idx = edges[:, (1)].contiguous()
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:H + Dout]
        new_o_vecs = new_t_vecs[:, H + Dout:2 * H + Dout]
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)
        if self.pooling == 'avg':
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
        new_obj_vecs = self.net2(pooled_obj_vecs)
        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'pooling': pooling, 'mlp_normalization': mlp_normalization}
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs


class Unflatten(nn.Module):

    def __init__(self, size):
        super(Unflatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(*self.size)

    def __repr__(self):
        size_str = ', '.join('%d' % d for d in self.size)
        return 'Unflatten(%s)' % size_str


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VectorPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.vectors = {}

    def query(self, objs, vectors):
        if self.pool_size == 0:
            return vectors
        return_vectors = []
        for obj, vector in zip(objs, vectors):
            obj = obj.item()
            vector = vector.cpu().clone().detach()
            if obj not in self.vectors:
                self.vectors[obj] = []
            obj_pool_size = len(self.vectors[obj])
            if obj_pool_size == 0:
                return_vectors.append(vector)
                self.vectors[obj].append(vector)
            elif obj_pool_size < self.pool_size:
                random_id = random.randint(0, obj_pool_size - 1)
                self.vectors[obj].append(vector)
                return_vectors.append(self.vectors[obj][random_id])
            else:
                random_id = random.randint(0, obj_pool_size - 1)
                tmp = self.vectors[obj][random_id]
                self.vectors[obj][random_id] = vector
                return_vectors.append(tmp)
        return_vectors = torch.stack(return_vectors)
        return return_vectors


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'conditional':
        norm_layer = functools.partial(ConditionalBatchNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    assert torch.cuda.is_available()
    netG
    netG.apply(weights_init)
    return netG


def mask_net(dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
        layers.append(Interpolate(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.ReLU())
        cur_size *= 2
    if cur_size != mask_size:
        raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)
    boxes = boxes.view(O, 4, 1, 1)
    x0, y0 = boxes[:, (0)], boxes[:, (1)]
    ww, hh = boxes[:, (2)] - x0, boxes[:, (3)] - y0
    X = torch.linspace(0, 1, steps=W).view(1, 1, W)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1)
    X = (X - x0) / ww
    Y = (Y - y0) / hh
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)
    grid = grid.mul(2).sub(1)
    return grid


def _pool_samples(samples, clean_mask_sampled, obj_to_img, pooling='sum'):
    """
    Input:
    - samples: FloatTensor of shape (O, D, H, W)
    - obj_to_img: LongTensor of shape (O,) with each element in the range
      [0, N) mapping elements of samples to output images

    Output:
    - pooled: FloatTensor of shape (N, D, H, W)
    """
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1
    obj_to_img_list = [i.item() for i in list(obj_to_img)]
    all_out = []
    if clean_mask_sampled is None:
        for i in range(N):
            start = obj_to_img_list.index(i)
            end = len(obj_to_img_list) - obj_to_img_list[::-1].index(i)
            all_out.append(torch.sum(samples[start:end, :, :, :], dim=0))
    else:
        _, d, h, w = samples.shape
        for i in range(N):
            start = obj_to_img_list.index(i)
            end = len(obj_to_img_list) - obj_to_img_list[::-1].index(i)
            mass = [torch.sum(samples[(j), :, :, :]).item() for j in range(start, end)]
            argsort = np.argsort(mass)
            result = torch.zeros((d, h, w), device=samples.device, dtype=samples.dtype)
            result_clean = torch.zeros((h, w), device=samples.device, dtype=samples.dtype)
            for j in argsort:
                masked_mask = (result_clean == 0).float() * (clean_mask_sampled[start + j, 0] > 0.5).float()
                result_clean += masked_mask
                result += samples[start + j] * masked_mask
            all_out.append(result)
    out = torch.stack(all_out)
    if pooling == 'avg':
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)
    return out


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum', test_mode=False):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - masks: Tensor of shape (O, M, M) giving binary masks for each object
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - H, W: Size of the output image.

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H
    grid = _boxes_to_grid(boxes, H, W)
    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)
    if test_mode:
        clean_mask_sampled = F.grid_sample(masks.float().view(O, 1, M, M), grid)
    else:
        clean_mask_sampled = None
    out = _pool_samples(sampled, clean_mask_sampled, obj_to_img, pooling=pooling)
    return out


class Model(nn.Module):

    def __init__(self, vocab, image_size=(64, 64), embedding_dim=128, gconv_dim=128, gconv_hidden_dim=512, gconv_pooling='avg', gconv_num_layers=5, mask_size=32, mlp_normalization='none', appearance_normalization='', activation='', n_downsample_global=4, box_dim=128, use_attributes=False, box_noise_dim=64, mask_noise_dim=64, pool_size=100, rep_size=32):
        super(Model, self).__init__()
        self.vocab = vocab
        self.image_size = image_size
        self.use_attributes = use_attributes
        self.box_noise_dim = box_noise_dim
        self.mask_noise_dim = mask_noise_dim
        self.object_size = 64
        self.fake_pool = VectorPool(pool_size)
        self.num_objs = len(vocab['object_to_idx'])
        self.num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(self.num_objs, embedding_dim)
        self.pred_embeddings = nn.Embedding(self.num_preds, embedding_dim)
        if use_attributes:
            attributes_dim = vocab['num_attributes']
        else:
            attributes_dim = 0
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {'input_dim': embedding_dim, 'attributes_dim': attributes_dim, 'output_dim': gconv_dim, 'hidden_dim': gconv_hidden_dim, 'pooling': gconv_pooling, 'mlp_normalization': mlp_normalization}
            self.gconv = GraphTripleConv(**gconv_kwargs)
        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {'input_dim': gconv_dim, 'hidden_dim': gconv_hidden_dim, 'pooling': gconv_pooling, 'num_layers': gconv_num_layers - 1, 'mlp_normalization': mlp_normalization}
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
        box_net_dim = 4
        self.box_dim = box_dim
        box_net_layers = [self.box_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
        self.g_mask_dim = gconv_dim + mask_noise_dim
        self.mask_net = mask_net(self.g_mask_dim, mask_size)
        self.repr_input = self.g_mask_dim
        rep_size = rep_size
        rep_hidden_size = 64
        repr_layers = [self.repr_input, rep_hidden_size, rep_size]
        self.repr_net = build_mlp(repr_layers, batch_norm=mlp_normalization)
        appearance_encoder_kwargs = {'vocab': vocab, 'arch': 'C4-64-2,C4-128-2,C4-256-2', 'normalization': appearance_normalization, 'activation': activation, 'padding': 'valid', 'vecs_size': self.g_mask_dim}
        self.image_encoder = AppearanceEncoder(**appearance_encoder_kwargs)
        netG_input_nc = self.num_objs + rep_size
        output_nc = 3
        ngf = 64
        n_blocks_global = 9
        norm = 'instance'
        self.layout_to_image = define_G(netG_input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm)

    def forward(self, gt_imgs, objs, triples, obj_to_img, boxes_gt=None, masks_gt=None, attributes=None, test_mode=False, use_gt_box=False, features=None):
        O, T = objs.size(0), triples.size(0)
        obj_vecs, pred_vecs = self.scene_graph_to_vectors(objs, triples, attributes)
        box_vecs, mask_vecs, scene_layout_vecs, wrong_layout_vecs = self.create_components_vecs(gt_imgs, boxes_gt, obj_to_img, objs, obj_vecs, features)
        boxes_pred = self.box_net(box_vecs)
        mask_scores = self.mask_net(mask_vecs.view(O, -1, 1, 1))
        masks_pred = mask_scores.squeeze(1).sigmoid()
        H, W = self.image_size
        if test_mode:
            boxes = boxes_gt if use_gt_box else boxes_pred
            masks = masks_gt if masks_gt is not None else masks_pred
            gt_layout = None
            pred_layout = masks_to_layout(scene_layout_vecs, boxes, masks, obj_to_img, H, W, test_mode=True)
            wrong_layout = None
            imgs_pred = self.layout_to_image(pred_layout)
        else:
            gt_layout = masks_to_layout(scene_layout_vecs, boxes_gt, masks_gt, obj_to_img, H, W, test_mode=False)
            pred_layout = masks_to_layout(scene_layout_vecs, boxes_gt, masks_pred, obj_to_img, H, W, test_mode=False)
            wrong_layout = masks_to_layout(wrong_layout_vecs, boxes_gt, masks_gt, obj_to_img, H, W, test_mode=False)
            imgs_pred = self.layout_to_image(gt_layout)
        return imgs_pred, boxes_pred, masks_pred, gt_layout, pred_layout, wrong_layout

    def scene_graph_to_vectors(self, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)
        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)
        if self.use_attributes:
            obj_vecs = torch.cat([obj_vecs, attributes], dim=1)
        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

    def create_components_vecs(self, imgs, boxes, obj_to_img, objs, obj_vecs, features):
        O = objs.size(0)
        box_vecs = obj_vecs
        mask_vecs = obj_vecs
        layout_noise = torch.randn((1, self.mask_noise_dim), dtype=mask_vecs.dtype, device=mask_vecs.device).repeat((O, 1)).view(O, self.mask_noise_dim)
        mask_vecs = torch.cat([mask_vecs, layout_noise], dim=1)
        if features is None:
            crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
            obj_repr = self.repr_net(self.image_encoder(crops))
        else:
            obj_repr = self.repr_net(mask_vecs)
            for ind, feature in enumerate(features):
                if feature is not None:
                    obj_repr[(ind), :] = feature
        one_hot_size = O, self.num_objs
        one_hot_obj = torch.zeros(one_hot_size, dtype=obj_repr.dtype, device=obj_repr.device)
        one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)
        layout_vecs = torch.cat([one_hot_obj, obj_repr], dim=1)
        wrong_objs_rep = self.fake_pool.query(objs, obj_repr)
        wrong_layout_vecs = torch.cat([one_hot_obj, wrong_objs_rep], dim=1)
        return box_vecs, mask_vecs, layout_vecs, wrong_layout_vecs

    def encode_scene_graphs(self, scene_graphs, rand=False):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:

        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }

        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.

        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.

        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
        if isinstance(scene_graphs, dict):
            scene_graphs = [scene_graphs]
        device = next(self.parameters()).device
        objs, triples, obj_to_img = [], [], []
        all_attributes = []
        all_features = []
        obj_offset = 0
        for i, sg in enumerate(scene_graphs):
            attributes = torch.zeros([len(sg['objects']) + 1, 25 + 10], dtype=torch.float, device=device)
            sg['objects'].append('__image__')
            sg['features'].append(sg['image_id'])
            image_idx = len(sg['objects']) - 1
            for j in range(image_idx):
                sg['relationships'].append([j, '__in_image__', image_idx])
            for obj in sg['objects']:
                obj_idx = self.vocab['object_to_idx'][str(self.vocab['object_name_to_idx'][obj])]
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            if self.features is not None:
                for obj_name, feat_num in zip(objs, sg['features']):
                    if feat_num == -1:
                        feat = self.features_one[obj_name][0]
                    else:
                        feat = self.features[obj_name][(min(feat_num, 99)), :]
                    feat = torch.from_numpy(feat).type(torch.float32)
                    all_features.append(feat)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            for i, size_attr in enumerate(sg['attributes']['size']):
                attributes[i, size_attr] = 1
            attributes[-1, 9] = 1
            for i, location_attr in enumerate(sg['attributes']['location']):
                attributes[i, location_attr + 10] = 1
            attributes[-1, 12 + 10] = 1
            obj_offset += len(sg['objects'])
            all_attributes.append(attributes)
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        attributes = torch.cat(all_attributes)
        features = all_features
        return objs, triples, obj_to_img, attributes, features

    def forward_json(self, scene_graphs):
        """ Convenience method that combines encode_scene_graphs and forward. """
        objs, triples, obj_to_img, attributes, features = self.encode_scene_graphs(scene_graphs)
        return self.forward(None, objs, triples, obj_to_img, attributes=attributes, test_mode=True, use_gt_box=False, features=features), objs


class InceptionScore(nn.Module):

    def __init__(self, cuda=True, batch_size=32, resize=False):
        super(InceptionScore, self).__init__()
        assert batch_size > 0
        self.resize = resize
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = 'cuda' if cuda else 'cpu'
        if not cuda and torch.cuda.is_available():
            None
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.up = Interpolate(size=(299, 299), mode='bilinear')
        self.clean()

    def clean(self):
        self.preds = np.zeros((0, 1000))

    def get_pred(self, x):
        if self.resize:
            x = self.up(x)
        x = self.inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def forward(self, imgs):
        preds_imgs = self.get_pred(imgs)
        self.preds = np.append(self.preds, preds_imgs, axis=0)

    def compute_score(self, splits=1):
        split_scores = []
        preds = self.preds
        N = self.preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits):(k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[(i), :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores), np.std(split_scores)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (InceptionScore,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGGLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_ashual_scene_generation(_paritybench_base):
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

