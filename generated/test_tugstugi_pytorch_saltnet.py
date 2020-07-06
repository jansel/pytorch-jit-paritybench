import sys
_module = sys.modules[__name__]
del sys
datasets = _module
salt_identification = _module
losses = _module
lovasz_losses = _module
models = _module
basenet = _module
inplace_abn = _module
abn = _module
bn = _module
functions = _module
oc_net = _module
unet = _module
swa = _module
test = _module
train = _module
transforms = _module
unet_transforms = _module
utils = _module
adamw = _module
lr_scheduler = _module
metrics = _module
rle = _module

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


import copy


from torch.utils.data import Dataset


import functools


import torch


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


import torch.nn as nn


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


import torchvision


import torch.nn.functional as functional


import torch.autograd as autograd


import torch.cuda.comm as comm


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import load


from torch import nn


from torch.nn import functional as F


import time


from torch.utils.data import DataLoader


from torchvision.transforms import *


from torch.utils.data import ConcatDataset


import torchvision.utils as vutils


import torch.optim as optim


import torch.backends.cudnn as cudnn


import math


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class DummyModule(nn.Module):

    def forward(self, x):
        return x


class MockModule(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.backbone = nn.ModuleList(layers)


class ABN(nn.BatchNorm2d):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        if activation not in ('leaky_relu', 'elu', 'none'):
            raise NotImplementedError(activation)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = super().forward(x)
        if self.activation == 'leaky_relu':
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == 'elu':
            return functional.elu(x, inplace=True)
        else:
            return x

    def extra_repr(self):
        rep = super().extra_repr()
        rep += ', activation={activation}'.format(**self.__dict__)
        if self.activation == 'leaky_relu':
            rep += ', slope={slope}'.format(**self.__dict__)
        return rep


ACT_LEAKY_RELU = 'leaky_relu'


ACT_ELU = 'elu'


ACT_NONE = 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABN(autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * count / (count - 1))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))
        dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = dweight if ctx.affine else None
        dbias = dbias if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var, extra, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        cls._parse_extra(ctx, extra)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        count = _count_samples(x) * (ctx.master_queue.maxsize + 1)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.is_master:
                means, vars = [mean.unsqueeze(0)], [var.unsqueeze(0)]
                for _ in range(ctx.master_queue.maxsize):
                    mean_w, var_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    means.append(mean_w.unsqueeze(0))
                    vars.append(var_w.unsqueeze(0))
                means = comm.gather(means)
                vars = comm.gather(vars)
                mean = means.mean(0)
                var = (vars + (mean - means) ** 2).mean(0)
                tensors = comm.broadcast_coalesced((mean, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((mean, var))
                mean, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * count / (count - 1))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            if ctx.is_master:
                edzs, eydzs = [edz], [eydz]
                for _ in range(len(ctx.worker_queues)):
                    edz_w, eydz_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    edzs.append(edz_w)
                    eydzs.append(eydz_w)
                edz = comm.reduce_add(edzs) / (ctx.master_queue.maxsize + 1)
                eydz = comm.reduce_add(eydzs) / (ctx.master_queue.maxsize + 1)
                tensors = comm.broadcast_coalesced((edz, eydz), [edz.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((edz, eydz))
                edz, eydz = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))
        dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = dweight if ctx.affine else None
        dbias = dbias if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra['is_master']
        if ctx.is_master:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queues = extra['worker_queues']
            ctx.worker_ids = extra['worker_ids']
        else:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queue = extra['worker_queue']


class SelfAttentionBlock2D(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0), ActivatedBatchNorm(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class BaseOC(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=(1,)):
        super().__init__()
        self.stages = nn.ModuleList([SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0), ActivatedBatchNorm(out_channels), nn.Dropout2d(dropout))

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class Decoder(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), ActivatedBatchNorm(middle_channels), BaseOC(in_channels=middle_channels, out_channels=middle_channels, key_channels=middle_channels // 2, value_channels=middle_channels // 2, dropout=0.2), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class ConcatPool2d(nn.Module):

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.ap = nn.AvgPool2d(kernel_size, stride)
        self.mp = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, size=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def darknet(pretrained):
    net = DarkNet()
    if pretrained:
        state_dict = torch.load('/media/data/model_zoo/coco/pytorch_yolov3.pth')
        net.load_state_dict(state_dict)
    n_pretrained = 3 if pretrained else 0
    return [net.model0, net.model1, net.model2], True, n_pretrained


MODEL_ZOO_URL = 'https://drontheimerstr.synology.me/model_zoo/'


MODEL_URLS = {'resnet50': {'voc': MODEL_ZOO_URL + 'SSDretina_resnet50_c21-1c85a349.pth', 'coco': MODEL_ZOO_URL + 'SSDretina_resnet50_c81-a584ead7.pth', 'oid': MODEL_ZOO_URL + 'SSDretina_resnet50_c501-06095077.pth'}, 'resnext101_32x4d': {'coco': MODEL_ZOO_URL + 'SSDretina_resnext101_32x4d_c81-fdb37546.pth'}}


def load_pretrained_weights(layers, name, dataset_name):
    state_dict = model_zoo.load_url(MODEL_URLS[name][dataset_name])
    mock_module = MockModule(layers)
    mock_module.load_state_dict(state_dict, strict=False)


def get_out_channels(layers):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, 'out_channels'):
        return layers.out_channels
    elif isinstance(layers, int):
        return layers
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, 'out_channels'):
                return layer.out_channels
            elif isinstance(layer, nn.Sequential):
                return get_out_channels(layer)
    raise RuntimeError('cant get_out_channels from {}'.format(layers))


def Sequential(*args):
    f = nn.Sequential(*args)
    f.out_channels = get_out_channels(args)
    return f


def resnet(name, pretrained):
    if name == 'resnet18':
        net_class = torchvision.models.resnet18
    elif name == 'resnet34':
        net_class = torchvision.models.resnet34
    elif name == 'resnet50':
        net_class = torchvision.models.resnet50
    elif name == 'resnet101':
        net_class = torchvision.models.resnet101
    elif name == 'resnet152':
        net_class = torchvision.models.resnet152
    imagenet_pretrained = pretrained == 'imagenet'
    resnet = net_class(pretrained=imagenet_pretrained)
    layer0 = Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer0[-1].out_channels = resnet.bn1.num_features

    def get_out_channels_from_resnet_block(layer):
        block = layer[-1]
        if isinstance(block, torchvision.models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, torchvision.models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError('unknown resnet block: {}'.format(block))
    resnet.layer1.out_channels = resnet.layer1[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer1)
    resnet.layer2.out_channels = resnet.layer2[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer2)
    resnet.layer3.out_channels = resnet.layer3[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer3)
    resnet.layer4.out_channels = resnet.layer4[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer4)
    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4], True, n_pretrained


def resnext(name, pretrained):
    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented
    resnext_features = resnext.features
    layer0 = [resnext_features[i] for i in range(4)]
    layer0 = nn.Sequential(*layer0)
    layer0.out_channels = layer0[-1].out_channels = 64
    layer1 = resnext_features[4]
    layer1.out_channels = layer1[-1].out_channels = 256
    layer2 = resnext_features[5]
    layer2.out_channels = layer2[-1].out_channels = 512
    layer3 = resnext_features[6]
    layer3.out_channels = layer3[-1].out_channels = 1024
    layer4 = resnext_features[7]
    layer4.out_channels = layer4[-1].out_channels = 2048
    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def replace_bn(bn, act=None):
    slop = 0.01
    if isinstance(act, nn.ReLU):
        activation = 'leaky_relu'
    elif isinstance(act, nn.LeakyReLU):
        activation = 'leaky_relu'
        slope = act.negative_slope
    elif isinstance(act, nn.ELU):
        activation = 'elu'
    else:
        activation = 'none'
    abn = ActivatedBatchNorm(num_features=bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=bn.affine, track_running_stats=bn.track_running_stats, activation=activation, slope=slop)
    abn.load_state_dict(bn.state_dict())
    return abn


def replace_bn_in_block(block):
    block.bn1 = replace_bn(block.bn1, block.relu)
    block.bn2 = replace_bn(block.bn2, block.relu)
    block.bn3 = replace_bn(block.bn3)
    block.relu = DummyModule()
    if block.downsample:
        block.downsample = replace_bn_in_sequential(block.downsample)
    return nn.Sequential(block, nn.ReLU(inplace=True))


def replace_bn_in_sequential(layer0, block=None):
    layer0_modules = []
    last_bn = None
    for n, m in layer0.named_children():
        if isinstance(m, nn.BatchNorm2d):
            last_bn = n, m
        else:
            activation = 'none'
            if last_bn:
                abn = replace_bn(last_bn[1], m)
                activation = abn.activation
                layer0_modules.append((last_bn[0], abn))
                last_bn = None
            if activation == 'none':
                if block and isinstance(m, block):
                    m = replace_bn_in_block(m)
                elif isinstance(m, nn.Sequential):
                    m = replace_bn_in_sequential(m, block)
                layer0_modules.append((n, m))
    if last_bn:
        abn = replace_bn(last_bn[1])
        layer0_modules.append((last_bn[0], abn))
    return nn.Sequential(OrderedDict(layer0_modules))


def se_net(name, pretrained):
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented
    layer0 = replace_bn_in_sequential(senet.layer0)
    block = senet.layer1[0].__class__
    layer1 = replace_bn_in_sequential(senet.layer1, block=block)
    layer1.out_channels = layer1[-1].out_channels = senet.layer1[-1].conv3.out_channels
    layer0.out_channels = layer0[-1].out_channels = senet.layer1[0].conv1.in_channels
    layer2 = replace_bn_in_sequential(senet.layer2, block=block)
    layer2.out_channels = layer2[-1].out_channels = senet.layer2[-1].conv3.out_channels
    layer3 = replace_bn_in_sequential(senet.layer3, block=block)
    layer3.out_channels = layer3[-1].out_channels = senet.layer3[-1].conv3.out_channels
    layer4 = replace_bn_in_sequential(senet.layer4, block=block)
    layer4.out_channels = layer4[-1].out_channels = senet.layer4[-1].conv3.out_channels
    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def ConvBnRelu(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(c, nn.BatchNorm2d(c.out_channels), nn.ReLU(inplace=True))


def ConvRelu(*args, **kwargs):
    return Sequential(nn.Conv2d(*args, **kwargs), nn.ReLU(inplace=True))


def vgg_base_extra(bn):
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    block = ConvBnRelu if bn else ConvRelu
    conv6 = block(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = block(1024, 1024, kernel_size=1)
    return [pool5, conv6, conv7]


def vgg(name, pretrained):
    if name == 'vgg11':
        net_class = torchvision.models.vgg11
    elif name == 'vgg13':
        net_class = torchvision.models.vgg13
    elif name == 'vgg16':
        net_class = torchvision.models.vgg16
    elif name == 'vgg19':
        net_class = torchvision.models.vgg19
    elif name == 'vgg11_bn':
        net_class = torchvision.models.vgg11_bn
    elif name == 'vgg13_bn':
        net_class = torchvision.models.vgg13_bn
    elif name == 'vgg16_bn':
        net_class = torchvision.models.vgg16_bn
    elif name == 'vgg19_bn':
        net_class = torchvision.models.vgg19_bn
    else:
        raise RuntimeError('unknown model {}'.format(name))
    imagenet_pretrained = pretrained == 'imagenet'
    vgg = net_class(pretrained=imagenet_pretrained)
    if name == 'vgg16':
        vgg.features[16].ceil_mode = True
    bn = name.endswith('bn')
    layers = []
    l = []
    for i in range(len(vgg.features) - 1):
        if isinstance(vgg.features[i], nn.MaxPool2d):
            layers.append(l)
            l = []
        l.append(vgg.features[i])
    l += vgg_base_extra(bn=bn)
    layers.append(l)
    block = ConvBnRelu if bn else ConvRelu
    layer5 = [block(1024, 256, 1, 1, 0), block(256, 512, 3, 2, 1)]
    layers.append(layer5)
    layers = [Sequential(*l) for l in layers]
    n_pretrained = 4 if imagenet_pretrained else 0
    return layers, bn, n_pretrained


def create_basenet(name, pretrained):
    """
    Parameters
    ----------
    name: model name
    pretrained: dataset name

    Returns
    -------
    list of modules, is_batchnorm, num_of_pretrained_module
    """
    if name.startswith('vgg'):
        layers, bn, n_pretrained = vgg(name, pretrained)
    elif name.startswith('resnet'):
        layers, bn, n_pretrained = resnet(name, pretrained)
    elif name.startswith('resnext'):
        layers, bn, n_pretrained = resnext(name, pretrained)
    elif name.startswith('se'):
        layers, bn, n_pretrained = se_net(name, pretrained)
    elif name == 'darknet':
        layers, bn, n_pretrained = darknet(pretrained)
    else:
        raise NotImplemented(name)
    if pretrained in ('coco', 'oid'):
        load_pretrained_weights(layers, name, pretrained)
        n_pretrained = len(layers)
    return layers, bn, n_pretrained


def upsample(size=None, scale_factor=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)


class UNet(nn.Module):

    def __init__(self, basenet='vgg11', num_filters=16, pretrained='imagenet'):
        super().__init__()
        net, bn, n_pretrained = create_basenet(basenet, pretrained)
        if basenet.startswith('vgg'):
            self.encoder1 = net[0]
        else:
            self.encoder1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), net[0])
            self.encoder1.out_channels = net[0].out_channels
        self.encoder2 = net[1]
        self.encoder3 = net[2]
        self.encoder4 = net[3]
        context_channels = num_filters * 8 * 4
        self.encoder5 = nn.Sequential(net[4], nn.Conv2d(net[4].out_channels, context_channels, kernel_size=3, stride=1, padding=1), ActivatedBatchNorm(context_channels, activation='none'), BaseOC(in_channels=context_channels, out_channels=context_channels, key_channels=context_channels // 2, value_channels=context_channels // 2, dropout=0.05))
        self.encoder5.out_channels = context_channels
        self.fuse_image = nn.Sequential(nn.Linear(512, 32), nn.ReLU(inplace=True))
        self.logit_image = nn.Sequential(nn.Linear(32, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 8 * 2, num_filters * 8)
        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        if basenet.startswith('vgg'):
            self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)
            self.decoder1 = nn.Sequential(nn.Conv2d(self.encoder1.out_channels + num_filters, num_filters, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        else:
            self.decoder2 = nn.Sequential(nn.Conv2d(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters * 2 * 2), nn.Conv2d(num_filters * 2 * 2, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters))
            self.decoder1 = Decoder(self.encoder1.out_channels + num_filters, num_filters * 2, num_filters)
        self.logit = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(96, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))
        self.fuse_pixel = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters * (8 + 4 + 2 + 1 + 1), 64, kernel_size=1, padding=0))
        self.logit_pixel5 = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters * 8, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))
        self.logit_pixel4 = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters * 4, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))
        self.logit_pixel3 = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))
        self.logit_pixel2 = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))
        self.logit_pixel1 = nn.Sequential(nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), ActivatedBatchNorm(num_filters), nn.Conv2d(num_filters, 1, kernel_size=1))

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, (0), :, :] -= mean[0]
        x[:, (0), :, :] /= std[0]
        x[:, (1), :, :] -= mean[1]
        x[:, (1), :, :] /= std[1]
        x[:, (2), :, :] -= mean[2]
        x[:, (2), :, :] /= std[2]
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        c = self.center(self.pool(e5))
        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(torch.cat((d3, e2), 1))
        d1 = self.decoder1(d2, e1)
        d1_size = d1.size()[2:]
        upsampler = upsample(size=d1_size)
        u5 = upsampler(d5)
        u4 = upsampler(d4)
        u3 = upsampler(d3)
        u2 = upsampler(d2)
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        fuse_pixel = self.fuse_pixel(d)
        logit_pixel = self.logit_pixel1(d1), self.logit_pixel2(u2), self.logit_pixel3(u3), self.logit_pixel4(u4), self.logit_pixel5(u5)
        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        e = F.dropout(e, p=0.5, training=self.training)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image).view(-1)
        logit = self.logit(torch.cat([fuse_pixel, F.upsample(fuse_image.view(batch_size, -1, 1, 1), scale_factor=128, mode='nearest')], 1))
        return logit, logit_pixel, logit_image


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveConcatPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tugstugi_pytorch_saltnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

