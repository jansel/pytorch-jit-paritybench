import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main_kinetics = _module
main_something = _module
models = _module
ops = _module
basic_ops = _module
utils = _module
opts = _module
resnet_TSM = _module
thop = _module
count_hooks = _module
profile = _module
transforms = _module
tsm_util = _module

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


import torch.utils.data as data


import numpy as np


from numpy.random import randint


import time


import torch


import torchvision


import torch.nn.parallel


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torch.optim


from torch.nn.utils import clip_grad_norm_


import math


import torch.utils.model_zoo as model_zoo


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torch import nn


from sklearn.metrics import confusion_matrix


import torch.nn as nn


import torch as tr


import logging


from torch.nn.modules.conv import _ConvNd


import random


import numbers


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return ret_img_group, label

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [(self.input_size[1] if abs(x - self.input_size[1]) < 3 else x) for x in crop_sizes]
        crop_w = [(self.input_size[0] if abs(x - self.input_size[0]) < 3 else x) for x in crop_sizes]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))
            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, selective_flip=True, is_flow=False):
        self.is_flow = is_flow
        self.class_LeftRight = [86, 87, 93, 94, 166, 167] if selective_flip else []

    def __call__(self, img_tuple, is_flow=False):
        img_group, label = img_tuple
        v = random.random()
        if label not in self.class_LeftRight and v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret, label
        else:
            return img_tuple


class TSN(nn.Module):

    def __init__(self, num_class, num_segments, pretrained_parts, modality, base_model='resnet101', dataset='something', new_length=None, consensus_type='avg', before_softmax=True, dropout=0.8, fc_lr5=True, crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.pretrained_parts = pretrained_parts
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        self.dataset = dataset
        self.fc_lr5 = fc_lr5
        if not before_softmax and consensus_type != 'avg':
            raise ValueError('Only avg consensus can be used after Softmax')
        if new_length is None:
            self.new_length = 1 if modality == 'RGB' else 1
        else:
            self.new_length = new_length
        None
        if base_model == 'TSM':
            self.base_model = resnet50(True, shift='TSM', num_segments=num_segments)
            self.base_model.last_layer_name = 'fc1'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            feature_dim = self._prepare_tsn(num_class)
        elif base_model == 'MS':
            self.base_model = resnet18(True, shift='TSM', num_segments=num_segments, flow_estimation=1)
            self.base_model.last_layer_name = 'fc1'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = self._prepare_tsn(num_class)
        else:
            self._prepare_base_model(base_model)
            feature_dim = self._prepare_tsn(num_class)
        """
        # zc: print "NN variable name"
        zc_params = self.base_model.state_dict()
        for zc_k in zc_params.items():
            print(zc_k)

        # zc: print "Specified layer's weight and bias"
        print(zc_params['conv1_7x7_s2.weight'])
        print(zc_params['conv1_7x7_s2.bias'])
        """
        if self.modality == 'Flow':
            None
            self.base_model = self._construct_flow_model(self.base_model)
            None
        elif self.modality == 'RGBDiff':
            None
            self.base_model = self._construct_diff_model(self.base_model)
            None
        self.consensus = ConsensusModule(consensus_type)
        if not self.before_softmax:
            self.softmax = nn.Softmax()
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_channels
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
        std = 0.001
        if self.new_fc is None:
            xavier_uniform_(getattr(self.base_model, self.base_model.last_layer_name).weight)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'C3DRes18':
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc8'
            self.input_size = 112
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            None

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, dataset):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError('New atomic module type: {}. Need to give it a learning policy'.format(type(m)))
        return [{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1, 'name': 'first_conv_weight'}, {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'name': 'first_conv_bias'}, {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'normal_weight'}, {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'normal_bias'}, {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'BN scale/shift'}, {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1, 'name': 'custom_ops'}, {'params': lr5_weight, 'lr_mult': 5 if self.dataset == 'kinetics' else 1, 'decay_mult': 1, 'name': 'lr5_weight'}, {'params': lr10_bias, 'lr_mult': 10 if self.dataset == 'kinetics' else 2, 'decay_mult': 0, 'name': 'lr10_bias'}]

    def get_optim_policies_BN2to1D(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        last_conv_weight = []
        last_conv_bias = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                last_conv_weight.append(ps[0])
                if len(ps) == 2:
                    last_conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError('New atomic module type: {}. Need to give it a learning policy'.format(type(m)))
        return [{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1, 'name': 'first_conv_weight'}, {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'name': 'first_conv_bias'}, {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'normal_weight'}, {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'normal_bias'}, {'params': last_conv_weight, 'lr_mult': 5, 'decay_mult': 1, 'name': 'last_conv_weight'}, {'params': last_conv_bias, 'lr_mult': 10, 'decay_mult': 0, 'name': 'last_conv_bias'}, {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'BN scale/shift'}]

    def forward(self, input, temperature):
        sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)
        if self.base_model_name == 'C3DRes18':
            before_permute = input.view((-1, sample_len) + input.size()[-2:])
            input_var = torch.transpose(before_permute.view((-1, self.num_segments) + before_permute.size()[1:]), 1, 2)
        elif 'Res3D' in self.base_model_name:
            before_permute = input.view((-1, sample_len) + input.size()[-2:])
            input_var = torch.transpose(before_permute.view((-1, self.num_segments) + before_permute.size()[1:]), 1, 2)
        elif self.base_model_name in ['I3D', 'I3D_flow']:
            before_permute = input.view((-1, sample_len) + input.size()[-2:])
            input_var = torch.transpose(before_permute.view((-1, self.num_segments) + before_permute.size()[1:]), 1, 2)
        else:
            input_var = input.view((-1, sample_len) + input.size()[-2:])
        base_out = self.base_model(input_var, temperature)
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            if 'flow' in self.base_model_name:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(3).squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ['RGB', 'RGBDiff'] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()
        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return new_data

    def _construct_flow_model(self, base_model):
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()), 1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]
        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66]), GroupRandomHorizontalFlip(selective_flip=True, is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)])


class Identity(torch.nn.Module):

    def forward(self, input):
        return input


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder = remainder
        self.num_segments = num_segments

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


def tsm(tensor, duration, version='zero'):
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 8, size[1] // 8, 3 * size[1] // 4], dim=2)
    if version == 'zero':
        pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
    elif version == 'circulant':
        pre_tensor = torch.cat((pre_tensor[:, -1:, ...], pre_tensor[:, :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:, 1:, ...], post_tensor[:, :1, ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder = remainder
        self.num_segments = num_segments

    def forward(self, x):
        identity = x
        out = tsm(x, self.num_segments, 'zero')
        out = self.conv1(out)
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remainder = remainder
        self.num_segments = num_segments

    def forward(self, x):
        identity = x
        out = tsm(x, self.num_segments, 'zero')
        out = self.conv1(out)
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


class Matching_layer_scs(nn.Module):

    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(Matching_layer_scs, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch
        self.correlation_sampler = SpatialCorrelationSampler(ks, patch, stride, pad, patch_dilation)

    def L2normalize(self, x, d=1):
        eps = 1e-06
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** 0.5
        return x / norm

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        corr = self.correlation_sampler(feature1, feature2)
        corr = corr.view(b, self.patch * self.patch, h1 * w1)
        corr = self.relu(corr)
        return corr


class Matching_layer_mm(nn.Module):

    def __init__(self, patch):
        super(Matching_layer_mm, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch

    def L2normalize(self, x, d=1):
        eps = 1e-06
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** 0.5
        return x / norm

    def corr_abs_to_rel(self, corr, h, w):
        max_d = self.patch // 2
        b, c, s = corr.size()
        corr = corr.view(b, h, w, h, w)
        w_diag = tr.zeros((b, h, h, self.patch, w), device='cuda')
        for i in range(max_d + 1):
            if i == 0:
                w_corr_offset = tr.diagonal(corr, offset=0, dim1=2, dim2=4)
                w_diag[:, :, :, max_d] = w_corr_offset
            else:
                w_corr_offset_pos = tr.diagonal(corr, offset=i, dim1=2, dim2=4)
                w_corr_offset_pos = F.pad(w_corr_offset_pos, (i, 0))
                w_diag[:, :, :, max_d - i] = w_corr_offset_pos
                w_corr_offset_neg = tr.diagonal(corr, offset=-i, dim1=2, dim2=4)
                w_corr_offset_neg = F.pad(w_corr_offset_neg, (0, i))
                w_diag[:, :, :, max_d + i] = w_corr_offset_neg
        hw_diag = tr.zeros((b, self.patch, w, self.patch, h), device='cuda')
        for i in range(max_d + 1):
            if i == 0:
                h_corr_offset = tr.diagonal(w_diag, offset=0, dim1=1, dim2=2)
                hw_diag[:, :, :, max_d] = h_corr_offset
            else:
                h_corr_offset_pos = tr.diagonal(w_diag, offset=i, dim1=1, dim2=2)
                h_corr_offset_pos = F.pad(h_corr_offset_pos, (i, 0))
                hw_diag[:, :, :, max_d - i] = h_corr_offset_pos
                h_corr_offset_neg = tr.diagonal(w_diag, offset=-i, dim1=1, dim2=2)
                h_corr_offset_neg = F.pad(h_corr_offset_neg, (0, i))
                hw_diag[:, :, :, max_d + i] = h_corr_offset_neg
        hw_diag = hw_diag.permute(0, 3, 1, 4, 2).contiguous()
        hw_diag = hw_diag.view(-1, self.patch * self.patch, h * w)
        return hw_diag

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        feature1 = feature1.view(b, c, h1 * w1)
        feature2 = feature2.view(b, c, h2 * w2)
        corr = tr.bmm(feature2.transpose(1, 2), feature1)
        corr = corr.view(b, h2 * w2, h1 * w1)
        corr = self.corr_abs_to_rel(corr, h1, w1)
        corr = self.relu(corr)
        return corr


class Flow_refinement(nn.Module):

    def __init__(self, num_segments, expansion=1, pos=2):
        super(Flow_refinement, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.pos = pos
        self.out_channel = 64 * 2 ** (self.pos - 1) * self.expansion
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, groups=3, bias=False), nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, self.c1, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.c1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False), nn.BatchNorm2d(self.c1), nn.ReLU(), nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.c2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False), nn.BatchNorm2d(self.c2), nn.ReLU(), nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.c3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False), nn.BatchNorm2d(self.c3), nn.ReLU(), nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.out_channel), nn.ReLU())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res, match_v):
        if match_v is not None:
            x = tr.cat([x, match_v], dim=1)
        _, c, h, w = x.size()
        x = x.view(-1, self.num_segments - 1, c, h, w)
        x = tr.cat([x, x[:, -1:, :, :, :]], dim=1)
        x = x.view(-1, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res
        return x


class ResNet(nn.Module):

    def __init__(self, block, block2, layers, num_segments, flow_estimation, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.num_segments = num_segments
        self.flow_estimation = flow_estimation
        if flow_estimation:
            self.patch = 15
            self.patch_dilation = 1
            self.matching_layer = Matching_layer_scs(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)
            self.flow_refinement = Flow_refinement(num_segments=num_segments, expansion=block.expansion, pos=2)
            self.soft_argmax = nn.Softmax(dim=1)
            self.chnl_reduction = nn.Sequential(nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1], num_segments=num_segments, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_segments=num_segments, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv1d(512 * block.expansion, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def L2normalize(self, x, d=1):
        eps = 1e-06
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** 0.5
        return x / norm

    def apply_binary_kernel(self, match, h, w, region):
        x_line = tr.arange(w, dtype=tr.float).detach()
        y_line = tr.arange(h, dtype=tr.float).detach()
        x_kernel_1 = x_line.view(1, 1, 1, 1, w).expand(1, 1, w, h, w).detach()
        y_kernel_1 = y_line.view(1, 1, 1, h, 1).expand(1, h, 1, h, w).detach()
        x_kernel_2 = x_line.view(1, 1, w, 1, 1).expand(1, 1, w, h, w).detach()
        y_kernel_2 = y_line.view(1, h, 1, 1, 1).expand(1, h, 1, h, w).detach()
        ones = tr.ones(1).detach()
        zeros = tr.zeros(1).detach()
        eps = 1e-06
        kx = tr.where(tr.abs(x_kernel_1 - x_kernel_2) <= region, ones, zeros).detach()
        ky = tr.where(tr.abs(y_kernel_1 - y_kernel_2) <= region, ones, zeros).detach()
        kernel = kx * ky + eps
        kernel = kernel.view(1, h * w, h * w).detach()
        return match * kernel

    def apply_gaussian_kernel(self, corr, h, w, p, sigma=5):
        b, c, s = corr.size()
        x = tr.arange(p, dtype=tr.float).detach()
        y = tr.arange(p, dtype=tr.float).detach()
        idx = corr.max(dim=1)[1]
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()
        x = x.view(1, 1, p, 1, 1).expand(1, 1, p, h, w).detach()
        y = y.view(1, p, 1, 1, 1).expand(1, p, 1, h, w).detach()
        gauss_kernel = tr.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, p * p, h * w)
        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h, w, temperature=1, mode='softmax'):
        b, c, s = match.size()
        idx = tr.arange(h * w, dtype=tr.float32)
        idx_x = idx % w
        idx_x = idx_x.repeat(b, k, 1)
        idx_y = tr.floor(idx / w)
        idx_y = idx_y.repeat(b, k, 1)
        soft_idx_x = idx_x[:, :1]
        soft_idx_y = idx_y[:, :1]
        displacement = (self.patch - 1) / 2
        topk_value, topk_idx = tr.topk(match, k, dim=1)
        topk_value = topk_value.view(-1, k, h, w)
        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match * temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre
        smax = smax.view(b, self.patch, self.patch, h, w)
        x_kernel = tr.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=tr.float)
        y_kernel = tr.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=tr.float)
        x_mult = x_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)
        y_mult = y_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)
        smax_x = smax.sum(dim=1, keepdim=False)
        smax_y = smax.sum(dim=2, keepdim=False)
        flow_x = (smax_x * x_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w)
        flow_y = (smax_y * y_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w)
        flow_x = flow_x / (self.patch_dilation * displacement)
        flow_y = flow_y / (self.patch_dilation * displacement)
        return flow_x, flow_y, topk_value

    def _make_layer(self, block, planes, blocks, num_segments, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder = int(i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
        return nn.Sequential(*layers)

    def flow_computation(self, x, pos=2, temperature=100):
        x = self.chnl_reduction(x)
        size = x.size()
        x = x.view((-1, self.num_segments) + size[1:])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        k = 1
        temperature = temperature
        b, c, t, h, w = x.size()
        t = t - 1
        x_pre = x[:, :, :-1].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = x[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        match = self.matching_layer(x_pre, x_post)
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = tr.cat([u, v], dim=1).view(-1, 2 * k, h, w)
        return flow, confidence

    def forward(self, x, temperature):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.flow_estimation == 1:
            flow_1, match_v = self.flow_computation(x, temperature=temperature)
            x = self.flow_refinement(flow_1, x, match_v)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1, 1)
        x = self.fc1(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock2,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'num_segments': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_arunos728_MotionSqueeze(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

