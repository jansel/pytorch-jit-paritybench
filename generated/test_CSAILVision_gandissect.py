import sys
_module = sys.modules[__name__]
del sys
netdissect = _module
aceoptimize = _module
aceplotablate = _module
acesummarize = _module
actviz = _module
autoeval = _module
broden = _module
dissection = _module
easydict = _module
evalablate = _module
fullablate = _module
modelconfig = _module
nethook = _module
parallelfolder = _module
pidfile = _module
plotutil = _module
proggan = _module
progress = _module
runningstats = _module
sampler = _module
segdata = _module
segmenter = _module
segmodel = _module
models = _module
resnet = _module
resnext = _module
segviz = _module
server = _module
serverstate = _module
statedict = _module
allunitsample = _module
ganseg = _module
makesample = _module
upsegmodel = _module
models = _module
prroi_pool = _module
build = _module
functional = _module
prroi_pool = _module
test_prroi_pooling2d = _module
resnet = _module
resnext = _module
workerpool = _module
zdataset = _module
ipynb_drop_output = _module
plot_ace = _module
plot_confroom = _module
plot_doorins = _module
plot_pixablate = _module
plot_window = _module
setup = _module

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


import numbers


import numpy


from torchvision import transforms


from torch.utils.data import TensorDataset


from scipy.ndimage.morphology import binary_dilation


import re


from collections import OrderedDict


from torchvision.datasets.folder import default_loader


from scipy import ndimage


import types


import torchvision


from collections import defaultdict


import torch.utils.data as data


import itertools


import torch.nn as nn


import math


import random


from torch.utils.data.sampler import Sampler


from torchvision.transforms.functional import to_tensor


from torchvision.transforms.functional import normalize


from torch.utils.data import DataLoader


from collections.abc import MutableMapping


from collections.abc import Mapping


from scipy.io import savemat


import torch.nn.functional as F


import torch.autograd as ag


def make_matching_tensor(valuedict, name, data):
    """
    Converts `valuedict[name]` to be a tensor with the same dtype, device,
    and dimension count as `data`, and caches the converted tensor.
    """
    v = valuedict.get(name, None)
    if v is None:
        return None
    if not isinstance(v, torch.Tensor):
        v = torch.from_numpy(numpy.array(v))
        valuedict[name] = v
    if not v.device == data.device or not v.dtype == data.dtype:
        assert not v.requires_grad, '%s wrong device or type' % name
        v = v
        valuedict[name] = v
    if len(v.shape) < len(data.shape):
        assert not v.requires_grad, '%s wrong dimensions' % name
        v = v.view((1,) + tuple(v.shape) + (1,) * (len(data.shape) - len(v.shape) - 1))
        valuedict[name] = v
    return v


class InstrumentedModel(torch.nn.Module):
    """
    A wrapper for hooking, probing and intervening in pytorch Modules.
    Example usage:

    ```
    model = load_my_model()
    with inst as InstrumentedModel(model):
        inst.retain_layer(layername)
        inst.edit_layer(layername, 0.5, target_features)
        inst(inputs)
        original_features = inst.retained_layer(layername)
    ```
    """

    def __init__(self, model):
        super(InstrumentedModel, self).__init__()
        self.model = model
        self._retained = OrderedDict()
        self._ablation = {}
        self._replacement = {}
        self._hooked_layer = {}
        self._old_forward = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retain_layer(self, layername):
        """
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and retain its output each time the model is run.
        A pair (layername, aka) can be provided, and the aka will be used
        as the key for the retained value instead of the layername.
        """
        self.retain_layers([layername])

    def retain_layers(self, layernames):
        """
        Retains a list of a layers at once.
        """
        self.add_hooks(layernames)
        for layername in layernames:
            aka = layername
            if not isinstance(aka, str):
                layername, aka = layername
            if aka not in self._retained:
                self._retained[aka] = None

    def retained_features(self):
        """
        Returns a dict of all currently retained features.
        """
        return OrderedDict(self._retained)

    def retained_layer(self, aka=None, clear=False):
        """
        Retrieve retained data that was previously hooked by retain_layer.
        Call this after the model is run.  If clear is set, then the
        retained value will return and also cleared.
        """
        if aka is None:
            aka = next(self._retained.keys().__iter__())
        result = self._retained[aka]
        if clear:
            self._retained[aka] = None
        return result

    def edit_layer(self, layername, ablation=None, replacement=None):
        """
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and modify its output each time the model is run.
        The output of the layer will be modified to be a convex combination
        of the replacement and x interpolated according to the ablation, i.e.:
        `output = x * (1 - a) + (r * a)`.
        """
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if ablation is None and replacement is not None:
            ablation = 1.0
        self.add_hooks([(layername, aka)])
        self._ablation[aka] = ablation
        self._replacement[aka] = replacement

    def remove_edits(self, layername=None):
        """
        Removes edits at the specified layer, or removes edits at all layers
        if no layer name is specified.
        """
        if layername is None:
            self._ablation.clear()
            self._replacement.clear()
            return
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if aka in self._ablation:
            del self._ablation[aka]
        if aka in self._replacement:
            del self._replacement[aka]

    def add_hooks(self, layernames):
        """
        Sets up a set of layers to be hooked.

        Usually not called directly: use edit_layer or retain_layer instead.
        """
        needed = set()
        aka_map = {}
        for name in layernames:
            aka = name
            if not isinstance(aka, str):
                name, aka = name
            if self._hooked_layer.get(aka, None) != name:
                aka_map[name] = aka
                needed.add(name)
        if not needed:
            return
        for name, layer in self.model.named_modules():
            if name in aka_map:
                needed.remove(name)
                aka = aka_map[name]
                self._hook_layer(layer, name, aka)
        for name in needed:
            raise ValueError('Layer %s not found in model' % name)

    def _hook_layer(self, layer, layername, aka):
        """
        Internal method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        if aka in self._hooked_layer:
            raise ValueError('Layer %s already hooked' % aka)
        if layername in self._old_forward:
            raise ValueError('Layer %s already hooked' % layername)
        self._hooked_layer[aka] = layername
        self._old_forward[layername] = layer, aka, layer.__dict__.get('forward', None)
        editor = self
        original_forward = layer.forward

        def new_forward(self, *inputs, **kwargs):
            original_x = original_forward(*inputs, **kwargs)
            x = editor._postprocess_forward(original_x, aka)
            return x
        layer.forward = types.MethodType(new_forward, layer)

    def _unhook_layer(self, aka):
        """
        Internal method to remove a hook, restoring the original forward method.
        """
        if aka not in self._hooked_layer:
            return
        layername = self._hooked_layer[aka]
        layer, check, old_forward = self._old_forward[layername]
        assert check == aka
        if old_forward is None:
            if 'forward' in layer.__dict__:
                del layer.__dict__['forward']
        else:
            layer.forward = old_forward
        del self._old_forward[layername]
        del self._hooked_layer[aka]
        if aka in self._ablation:
            del self._ablation[aka]
        if aka in self._replacement:
            del self._replacement[aka]
        if aka in self._retained:
            del self._replacement[aka]

    def _postprocess_forward(self, x, aka):
        """
        The internal method called by the hooked layers after they are run.
        """
        if aka in self._retained:
            self._retained[aka] = x.detach()
        a = make_matching_tensor(self._ablation, aka, x)
        if a is not None:
            x = x * (1 - a)
            v = make_matching_tensor(self._replacement, aka, x)
            if v is not None:
                x += v * a
        return x

    def close(self):
        """
        Unhooks all hooked layers in the model.
        """
        for aka in list(self._old_forward.keys()):
            self._unhook_layer(aka)
        assert len(self._old_forward) == 0


class PixelNormLayer(nn.Module):

    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


class WScaleLayer(nn.Module):

    def __init__(self, size, fan_in, gain=numpy.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / numpy.sqrt(fan_in)
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(x_size[0], self.size, x_size[2], x_size[3])
        return x


class NormConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class DoubleResolutionLayer(nn.Module):

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x


class NormUpscaleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = DoubleResolutionLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class OutputConvBlock(nn.Module):

    def __init__(self, in_channels, tanh=False):
        super().__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0, bias=False)
        self.wscale = WScaleLayer(3, in_channels, gain=1)
        self.clamp = nn.Hardtanh() if tanh else lambda x: x

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.clamp(x)
        return x


class ProgressiveGenerator(nn.Sequential):

    def __init__(self, resolution=None, sizes=None, modify_sequence=None, output_tanh=False):
        """
        A pytorch progessive GAN generator that can be converted directly
        from either a tensorflow model or a theano model.  It consists of
        a sequence of convolutional layers, organized in pairs, with an
        upsampling and reduction of channels at every other layer; and
        then finally followed by an output layer that reduces it to an
        RGB [-1..1] image.

        The network can be given more layers to increase the output
        resolution.  The sizes argument indicates the fieature depth at
        each upsampling, starting with the input z: [input-dim, 4x4-depth,
        8x8-depth, 16x16-depth...].  The output dimension is 2 * 2**len(sizes)

        Some default architectures can be selected by supplying the
        resolution argument instead.

        The optional modify_sequence function can be used to transform the
        sequence of layers before the network is constructed.

        If output_tanh is set to True, the network applies a tanh to clamp
        the output to [-1,1] before output; otherwise the output is unclamped.
        """
        assert (resolution is None) != (sizes is None)
        if sizes is None:
            sizes = {(8): [512, 512, 512], (16): [512, 512, 512, 512], (32): [512, 512, 512, 512, 256], (64): [512, 512, 512, 512, 256, 128], (128): [512, 512, 512, 512, 256, 128, 64], (256): [512, 512, 512, 512, 256, 128, 64, 32], (1024): [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]}[resolution]
        sequence = []

        def add_d(layer, name=None):
            if name is None:
                name = 'layer%d' % (len(sequence) + 1)
            sequence.append((name, layer))
        add_d(NormConvBlock(sizes[0], sizes[1], kernel_size=4, padding=3))
        add_d(NormConvBlock(sizes[1], sizes[1], kernel_size=3, padding=1))
        for i, (si, so) in enumerate(zip(sizes[1:-1], sizes[2:])):
            add_d(NormUpscaleConvBlock(si, so, kernel_size=3, padding=1))
            add_d(NormConvBlock(so, so, kernel_size=3, padding=1))
        dim = 4 * 2 ** (len(sequence) // 2 - 1)
        add_d(OutputConvBlock(sizes[-1], tanh=output_tanh), name='output_%dx%d' % (dim, dim))
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class SegmentationModuleBase(nn.Module):

    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


class SegmentationModule(SegmentationModuleBase):

    def __init__(self, net_enc, net_dec, labeldata, loss_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_dict = nn.ModuleDict()
        if loss_scale is None:
            self.loss_scale = {'object': 1, 'part': 0.5, 'scene': 0.25, 'material': 1}
        else:
            self.loss_scale = loss_scale
        self.crit_dict['object'] = nn.NLLLoss(ignore_index=0)
        self.crit_dict['material'] = nn.NLLLoss(ignore_index=0)
        self.crit_dict['scene'] = nn.NLLLoss(ignore_index=-1)
        self.labeldata = labeldata
        object_to_num = {k: v for v, k in enumerate(labeldata['object'])}
        part_to_num = {k: v for v, k in enumerate(labeldata['part'])}
        self.object_part = {object_to_num[k]: [part_to_num[p] for p in v] for k, v in labeldata['object_part'].items()}
        self.object_with_part = sorted(self.object_part.keys())
        self.decoder.object_part = self.object_part
        self.decoder.object_with_part = self.object_with_part

    def forward(self, feed_dict, *, seg_size=None):
        if seg_size is None:
            if feed_dict['source_idx'] == 0:
                output_switch = {'object': True, 'part': True, 'scene': True, 'material': False}
            elif feed_dict['source_idx'] == 1:
                output_switch = {'object': False, 'part': False, 'scene': False, 'material': True}
            else:
                raise ValueError
            pred = self.decoder(self.encoder(feed_dict['img'], return_feature_maps=True), output_switch=output_switch)
            loss_dict = {}
            if pred['object'] is not None:
                loss_dict['object'] = self.crit_dict['object'](pred['object'], feed_dict['seg_object'])
            if pred['part'] is not None:
                part_loss = 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    part_loss += self.part_loss(pred['part'][idx_part], feed_dict['seg_part'], feed_dict['seg_object'], object_label, feed_dict['valid_part'][:, (idx_part)])
                loss_dict['part'] = part_loss
            if pred['scene'] is not None:
                loss_dict['scene'] = self.crit_dict['scene'](pred['scene'], feed_dict['scene_label'])
            if pred['material'] is not None:
                loss_dict['material'] = self.crit_dict['material'](pred['material'], feed_dict['seg_material'])
            loss_dict['total'] = sum([(loss_dict[k] * self.loss_scale[k]) for k in loss_dict.keys()])
            metric_dict = {}
            if pred['object'] is not None:
                metric_dict['object'] = self.pixel_acc(pred['object'], feed_dict['seg_object'], ignore_index=0)
            if pred['material'] is not None:
                metric_dict['material'] = self.pixel_acc(pred['material'], feed_dict['seg_material'], ignore_index=0)
            if pred['part'] is not None:
                acc_sum, pixel_sum = 0, 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    acc, pixel = self.part_pixel_acc(pred['part'][idx_part], feed_dict['seg_part'], feed_dict['seg_object'], object_label, feed_dict['valid_part'][:, (idx_part)])
                    acc_sum += acc
                    pixel_sum += pixel
                metric_dict['part'] = acc_sum.float() / (pixel_sum.float() + 1e-10)
            if pred['scene'] is not None:
                metric_dict['scene'] = self.pixel_acc(pred['scene'], feed_dict['scene_label'], ignore_index=-1)
            return {'metric': metric_dict, 'loss': loss_dict}
        else:
            output_switch = {'object': True, 'part': True, 'scene': True, 'material': True}
            pred = self.decoder(self.encoder(feed_dict['img'], return_feature_maps=True), output_switch=output_switch, seg_size=seg_size)
            return pred


class Resnet(nn.Module):

    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), SynchronizedBatchNorm2d(out_planes), nn.ReLU(inplace=True))


class C1BilinearDeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, inference=False, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class C1Bilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, inference=False, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinearDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class UPerNet(nn.Module):

    def __init__(self, nr_classes, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.0))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False), SynchronizedBatchNorm2d(fpn_dim), nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)
        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = nr_classes['scene'], nr_classes['object'], nr_classes['part'], nr_classes['material']
        self.scene_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.AdaptiveAvgPool2d(1), nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True))
        self.object_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True))
        self.part_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True))
        self.material_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True))

    def forward(self, conv_out, output_switch=None, seg_size=None):
        output_dict = {k: None for k in output_switch.keys()}
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = []
        for i in range(input_size[0]):
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(pool_scale(conv5, roi.detach()), (input_size[2], input_size[3]), mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        if output_switch['scene']:
            output_dict['scene'] = self.scene_head(f)
        if output_switch['object'] or output_switch['part'] or output_switch['material']:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)
                f = F.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
                f = conv_x + f
                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse()
            if output_switch['material']:
                output_dict['material'] = self.material_head(fpn_feature_list[0])
            if output_switch['object'] or output_switch['part']:
                output_size = fpn_feature_list[0].size()[2:]
                fusion_list = [fpn_feature_list[0]]
                for i in range(1, len(fpn_feature_list)):
                    fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode='bilinear', align_corners=False))
                fusion_out = torch.cat(fusion_list, 1)
                x = self.conv_fusion(fusion_out)
                if output_switch['object']:
                    output_dict['object'] = self.object_head(x)
                if output_switch['part']:
                    output_dict['part'] = self.part_head(x)
        if self.use_softmax:
            x = output_dict['scene']
            x = x.squeeze(3).squeeze(2)
            x = F.softmax(x, dim=1)
            output_dict['scene'] = x
            for k in ['object', 'material']:
                x = output_dict[k]
                x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
                x = F.softmax(x, dim=1)
                output_dict[k] = x
            x = output_dict['part']
            x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            part_pred_list, head = [], 0
            for idx_part, object_label in enumerate(self.object_with_part):
                n_part = len(self.object_part[object_label])
                _x = F.interpolate(x[:, head:head + n_part], size=seg_size, mode='bilinear', align_corners=False)
                _x = F.softmax(_x, dim=1)
                part_pred_list.append(_x)
                head += n_part
            output_dict['part'] = part_pred_list
        else:
            for k in ['object', 'scene', 'material']:
                if output_dict[k] is None:
                    continue
                x = output_dict[k]
                x = F.log_softmax(x, dim=1)
                if k == 'scene':
                    x = x.squeeze(3).squeeze(2)
                output_dict[k] = x
            if output_dict['part'] is not None:
                part_pred_list, head = [], 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    n_part = len(self.object_part[object_label])
                    x = output_dict['part'][:, head:head + n_part]
                    x = F.log_softmax(x, dim=1)
                    part_pred_list.append(x)
                    head += n_part
                output_dict['part'] = part_pred_list
        return output_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), SynchronizedBatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(GroupBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 2)
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
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, groups=32, num_classes=1000):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), SynchronizedBatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PrRoIPool2DFunction(ag.Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), 'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())
        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)
        features = features.contiguous()
        rois = rois.contiguous()
        params = pooled_height, pooled_width, spatial_scale
        if features.is_cuda:
            output = _prroi_pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None
        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = _prroi_pooling.prroi_pooling_backward_cuda(features, rois, output, grad_output, *ctx.params)
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = _prroi_pooling.prroi_pooling_coor_backward_cuda(features, rois, output, grad_output, *ctx.params)
        return grad_input, grad_coor, None, None, None


prroi_pool2d = PrRoIPool2DFunction.apply


class PrRoIPool2D(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DoubleResolutionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InstrumentedModel,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (NormConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormUpscaleConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OutputConvBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WScaleLayer,
     lambda: ([], {'size': 4, 'fan_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_CSAILVision_gandissect(_paritybench_base):
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

