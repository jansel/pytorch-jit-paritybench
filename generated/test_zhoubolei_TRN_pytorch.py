import sys
_module = sys.modules[__name__]
del sys
TRNmodule = _module
average_scores = _module
dataset = _module
datasets_video = _module
extract_frames = _module
fps_dem_trn = _module
main = _module
models = _module
ops = _module
basic_ops = _module
utils = _module
opts = _module
process_dataset = _module
process_dataset_charades = _module
test_models = _module
test_video = _module
transforms = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


import numpy as np


import torch.utils.data as data


from numpy.random import randint


import torchvision


import torchvision.datasets as datasets


import re


import functools


import torch.nn.parallel


import torch.optim


from torch.nn import functional as F


import time


import torch.backends.cudnn as cudnn


from torch.nn.utils import clip_grad_norm


from torch import nn


from torch.nn.init import normal


from torch.nn.init import constant


import math


from sklearn.metrics import confusion_matrix


import random


import numbers


class RelationModule(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        num_bottleneck = 512
        classifier = nn.Sequential(nn.ReLU(), nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck), nn.ReLU(), nn.Linear(num_bottleneck, self.num_class))
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames * self.img_feature_dim)
        input = self.classifier(input)
        return input


class RelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]
        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))
        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(nn.ReLU(), nn.Linear(scale * self.img_feature_dim, num_bottleneck), nn.ReLU(), nn.Linear(num_bottleneck, self.num_class))
            self.fc_fusion_scales += [fc_fusion]
        None

    def forward(self, input):
        act_all = input[:, (self.relations_scales[0][0]), :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, (self.relations_scales[scaleID][idx]), :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class RelationModuleMultiScaleWithClassifier(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]
        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))
        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(nn.ReLU(), nn.Linear(scale * self.img_feature_dim, num_bottleneck), nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(num_bottleneck, num_bottleneck), nn.ReLU(), nn.Dropout(p=0.6))
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        None
        None

    def forward(self, input):
        act_all = input[:, (self.relations_scales[0][0]), :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)
        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, (self.relations_scales[scaleID][idx]), :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


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

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return ret_img_group

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

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class TSN(nn.Module):

    def __init__(self, num_class, num_segments, modality, base_model='resnet101', new_length=None, consensus_type='avg', before_softmax=True, dropout=0.8, img_feature_dim=256, crop_num=1, partial_bn=True, print_spec=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        if not before_softmax and consensus_type != 'avg':
            raise ValueError('Only avg consensus can be used after Softmax')
        if new_length is None:
            self.new_length = 1 if modality == 'RGB' else 5
        else:
            self.new_length = new_length
        if print_spec == True:
            None
        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)
        if self.modality == 'Flow':
            None
            self.base_model = self._construct_flow_model(self.base_model)
            None
        elif self.modality == 'RGBDiff':
            None
            self.base_model = self._construct_diff_model(self.base_model)
            None
        if consensus_type in ['TRN', 'TRNmultiscale']:
            self.consensus = TRNmodule.return_TRN(consensus_type, self.img_feature_dim, self.num_segments, num_class)
        else:
            self.consensus = ConsensusModule(consensus_type)
        if not self.before_softmax:
            self.softmax = nn.Softmax()
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN', 'TRNmultiscale']:
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                self.new_fc = nn.Linear(feature_dim, num_class)
        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
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
        elif base_model == 'BNInception':
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model == 'InceptionV3':
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif 'inception' in base_model:
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
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

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
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
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError('New atomic module type: {}. Need to give it a learning policy'.format(type(m)))
        return [{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1, 'name': 'first_conv_weight'}, {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'name': 'first_conv_bias'}, {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'normal_weight'}, {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'normal_bias'}, {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'BN scale/shift'}]

    def forward(self, input):
        sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ['RGB', 'RGBDiff'] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()
        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, (x), :, :, :] = input_view[:, :, (x), :, :, :] - input_view[:, :, (x - 1), :, :, :]
            else:
                new_data[:, :, (x - 1), :, :, :] = input_view[:, :, (x), :, :, :] - input_view[:, :, (x - 1), :, :, :]
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
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66]), GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)])


class Identity(torch.nn.Module):

    def forward(self, input):
        return input


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RelationModule,
     lambda: ([], {'img_feature_dim': 4, 'num_frames': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 16])], {}),
     True),
    (RelationModuleMultiScale,
     lambda: ([], {'img_feature_dim': 4, 'num_frames': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RelationModuleMultiScaleWithClassifier,
     lambda: ([], {'img_feature_dim': 4, 'num_frames': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_zhoubolei_TRN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

