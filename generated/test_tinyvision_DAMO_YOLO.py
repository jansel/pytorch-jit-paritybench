import sys
_module = sys.modules[__name__]
del sys
damoyolo_tinynasL20_T = _module
damoyolo_tinynasL25_S = _module
damoyolo_tinynasL35_M = _module
damo = _module
apis = _module
detector_inference = _module
detector_inference_trt = _module
detector_trainer = _module
augmentations = _module
box_level_augs = _module
color_augs = _module
gaussian_maps = _module
geometric_augs = _module
scale_aware_aug = _module
base_models = _module
backbones = _module
tinynas_csp = _module
tinynas_res = _module
atss_assigner = _module
bbox_calculator = _module
end2end = _module
ops = _module
ota_assigner = _module
utils = _module
weight_init = _module
heads = _module
zero_head = _module
distill_loss = _module
gfocal_loss = _module
necks = _module
giraffe_config = _module
giraffe_fpn_btn = _module
config = _module
base = _module
paths_catalog = _module
dataset = _module
build = _module
collate_batch = _module
datasets = _module
coco = _module
evaluation = _module
coco_eval = _module
mosaic_wrapper = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
transforms = _module
tta_aug = _module
detector = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
image_list = _module
boxes = _module
checkpoint = _module
debug_utils = _module
demo_utils = _module
dist = _module
imports = _module
logger = _module
metric = _module
model_utils = _module
timer = _module
visualize = _module
setup = _module
converter = _module
eval = _module
onnx_inference = _module
torch_inference = _module
train = _module
trt_eval = _module
trt_inference = _module

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


import torch


import numpy as np


import math


import random


import time


from copy import deepcopy


import torch.nn as nn


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.nn.functional as F


import copy


import torchvision.transforms as transforms


import warnings


from functools import partial


import torch.distributed as dist


import functools


import torch.utils.data


from torchvision.datasets.coco import CocoDetection


from collections import OrderedDict


from torch.utils.data.sampler import Sampler


import itertools


from torch.utils.data.sampler import BatchSampler


from torchvision.transforms import functional as F


import torchvision


from torch import distributed as dist


from collections import defaultdict


from collections import deque


import re


from torch.utils.cpp_extension import CppExtension


from torch import nn


class ConvKXBN(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, (kernel_size - 1) // 2, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()
    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


class ConvKXBNRELU(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """Basic cell for rep-style block, including conv and bn"""
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, act='relu', norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        """Forward process"""
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ResConvBlock(nn.Module):

    def __init__(self, in_c, out_c, btn_c, kernel_size, stride, act='silu', reparam=False, block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=kernel_size, stride=1)
        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c, out_c, kernel_size, stride, act='identity')
        self.activation_function = get_activation(act)
        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class CSPStem(nn.Module):

    def __init__(self, in_c, out_c, btn_c, stride, kernel_size, num_blocks, act='silu', reparam=False, block_type='k1kx'):
        super(CSPStem, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.stride = stride
        if self.stride == 2:
            self.num_blocks = num_blocks - 1
        else:
            self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.act = act
        self.block_type = block_type
        out_c = out_c // 2
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            if self.stride == 1 and block_id == 0:
                in_c = in_c // 2
            else:
                in_c = out_c
            the_block = ResConvBlock(in_c, out_c, btn_c, kernel_size, stride=1, act=act, reparam=reparam, block_type=block_type)
            self.block_list.append(the_block)

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module


class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, bias=False, act='silu', norm='bn', reparam=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SuperResStem(nn.Module):

    def __init__(self, in_c, out_c, btn_c, kernel_size, stride, num_blocks, with_spp=False, act='silu', reparam=False, block_type='k1kx'):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(in_channels, out_channels, btn_c, this_kernel_size, this_stride, act=act, reparam=reparam, block_type=block_type)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):

    def __init__(self, structure_info=None, out_indices=[2, 4, 5], with_spp=False, use_focus=False, act='silu', reparam=False):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()
        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
                    the_block = Focus(block_info['in'], block_info['out'], block_info['k'], act=act)
                else:
                    the_block = ConvKXBNRELU(block_info['in'], block_info['out'], block_info['k'], block_info['s'], act=act)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'], block_info['out'], block_info['btn'], block_info['k'], block_info['s'], block_info['L'], spp, act=act, reparam=reparam, block_type='k1kx')
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'], block_info['out'], block_info['btn'], block_info['k'], block_info['s'], block_info['L'], spp, act=act, reparam=reparam, block_type='kxkx')
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


class CSPWrapper(nn.Module):

    def __init__(self, convstem, act='relu', reparam=False, with_spp=False):
        super(CSPWrapper, self).__init__()
        self.with_spp = with_spp
        if isinstance(convstem, tuple):
            in_c = convstem[0].in_channels
            out_c = convstem[-1].out_channels
            hidden_dim = convstem[0].out_channels // 2
            _convstem = nn.ModuleList()
            for modulelist in convstem:
                for layer in modulelist.block_list:
                    _convstem.append(layer)
        else:
            in_c = convstem.in_channels
            out_c = convstem.out_channels
            hidden_dim = out_c // 2
            _convstem = convstem.block_list
        self.convstem = nn.ModuleList()
        for layer in _convstem:
            self.convstem.append(layer)
        self.act = get_activation(act)
        self.downsampler = ConvKXBNRELU(in_c, hidden_dim * 2, 3, 2, act=self.act)
        if self.with_spp:
            self.spp = SPPBottleneck(hidden_dim * 2, hidden_dim * 2)
        if len(self.convstem) > 0:
            self.conv_start = ConvKXBNRELU(hidden_dim * 2, hidden_dim, 1, 1, act=self.act)
            self.conv_shortcut = ConvKXBNRELU(hidden_dim * 2, out_c // 2, 1, 1, act=self.act)
            self.conv_fuse = ConvKXBNRELU(out_c, out_c, 1, 1, act=self.act)

    def forward(self, x):
        x = self.downsampler(x)
        if self.with_spp:
            x = self.spp(x)
        if len(self.convstem) > 0:
            shortcut = self.conv_shortcut(x)
            x = self.conv_start(x)
            for block in self.convstem:
                x = block(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_fuse(x)
        return x


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class=torch.tensor([100]), iou_threshold=torch.tensor([0.45]), score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.max_obj = torch.tensor([max_obj])
        self.iou_threshold = torch.tensor([iou_thres])
        self.score_threshold = torch.tensor([score_thres])
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=torch.float32, device=self.device)

    def forward(self, score, box):
        batch, anchors, _ = score.shape
        nms_box = box @ self.convert_matrix
        nms_score = score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nms_box, nms_score, self.max_obj, self.iou_threshold, self.score_threshold)
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)
        selected_box = nms_box[batch_inds, box_inds, ...]
        dets = torch.cat([selected_box, selected_score], dim=1)
        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)
        batch_template = torch.arange(0, batch, dtype=batch_inds.dtype, device=batch_inds.device)
        batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1), batched_dets.new_zeros(1))
        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)
        batched_labels = batched_labels.where(batch_inds == batch_template.unsqueeze(1), batched_labels.new_ones(1) * -1)
        N = batched_dets.shape[0]
        batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
        batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
        topk_batch_inds = torch.arange(batch, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]
        det_boxes, det_scores = batched_dets.split((4, 1), -1)
        det_scores = det_scores.squeeze(-1)
        num_det = (det_scores > 0).sum(1, keepdim=True)
        return num_det, det_boxes, det_scores, det_classes


class TRT7_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(ctx, boxes, scores, plugin_version='1', shareLocation=1, backgroundLabelId=-1, numClasses=80, topK=1000, keepTopK=100, scoreThreshold=0.25, iouThreshold=0.45, isNormalized=0, clipBoxes=0, scoreBits=16, caffeSemantics=1):
        batch_size, num_boxes, numClasses = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses, (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes, scores, plugin_version='1', shareLocation=1, backgroundLabelId=-1, numClasses=80, topK=1000, keepTopK=100, scoreThreshold=0.25, iouThreshold=0.45, isNormalized=0, clipBoxes=0, scoreBits=16, caffeSemantics=1):
        out = g.op('TRT::BatchedNMSDynamic_TRT', boxes, scores, shareLocation_i=shareLocation, plugin_version_s=plugin_version, backgroundLabelId_i=backgroundLabelId, numClasses_i=numClasses, topK_i=topK, keepTopK_i=keepTopK, scoreThreshold_f=scoreThreshold, iouThreshold_f=iouThreshold, isNormalized_i=isNormalized, clipBoxes_i=clipBoxes, scoreBits_i=scoreBits, caffeSemantics_i=caffeSemantics, outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT7(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.shareLocation = 1
        self.backgroundLabelId = -1
        self.numClasses = 80
        self.topK = 1000
        self.keepTopK = max_obj
        self.scoreThreshold = score_thres
        self.iouThreshold = iou_thres
        self.isNormalized = 0
        self.clipBoxes = 0
        self.scoreBits = 16
        self.caffeSemantics = 1
        self.plugin_version = '1'
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=torch.float32, device=self.device)

    def forward(self, score, box):
        box = box.unsqueeze(2)
        self.numClasses = int(score.shape[2])
        num_det, det_boxes, det_scores, det_classes = TRT7_NMS.apply(box, score, self.plugin_version, self.shareLocation, self.backgroundLabelId, self.numClasses, self.topK, self.keepTopK, self.scoreThreshold, self.iouThreshold, self.isNormalized, self.clipBoxes, self.scoreBits, self.caffeSemantics)
        return num_det, det_boxes, det_scores, det_classes.int()


class TRT8_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(ctx, boxes, scores, background_class=-1, box_coding=1, iou_threshold=0.45, max_output_boxes=100, plugin_version='1', score_activation=0, score_threshold=0.25):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes, scores, background_class=-1, box_coding=1, iou_threshold=0.45, max_output_boxes=100, plugin_version='1', score_activation=0, score_threshold=0.25):
        out = g.op('TRT::EfficientNMS_TRT', boxes, scores, background_class_i=background_class, box_coding_i=box_coding, iou_threshold_f=iou_threshold, max_output_boxes_i=max_output_boxes, plugin_version_s=plugin_version, score_activation_i=score_activation, score_threshold_f=score_threshold, outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT8(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, score, box):
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(box, score, self.background_class, self.box_coding, self.iou_threshold, self.max_obj, self.plugin_version, self.score_activation, self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None, ort=False, trt_version=7, with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model
        TRT = ONNX_TRT8 if trt_version >= 8 else ONNX_TRT7
        self.patch_model = ONNX_ORT if ort else TRT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:, [2, 1, 0], ...]
            x = x * (1 / 255)
        x = self.model(x)
        x = self.end2end(x[0], x[1])
        return x


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class BasicBlock_3x3_Reverse(nn.Module):

    def __init__(self, ch_in, ch_hidden_ratio, ch_out, act='relu', shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):

    def __init__(self, ch_in, ch_out, k, pool_size, act='swish'):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        y = self.conv(y)
        return y


class CSPStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_hidden_ratio, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(str(i), BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, act=act, shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x


class AssignResult(object):
    """Stores assignments between predicted and truth boxes.
    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {'num_gts': self.num_gts, 'num_preds': self.num_preds, 'gt_inds': self.gt_inds, 'max_overlaps': self.max_overlaps, 'labels': self.labels}
        basic_info.update(self._extra_properties)
        return basic_info

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.
        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state
        Returns:
            :obj:`AssignResult`: Randomly generated assign results.
        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        rng = demodata.ensure_rng(kwargs.get('rng', None))
        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)
        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)
        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np
            max_overlaps = torch.from_numpy(rng.rand(num_preds))
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))
            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()
            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True
            is_ignore = torch.from_numpy(rng.rand(num_preds) < p_ignore) & is_assigned
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]
            gt_inds = torch.from_numpy(rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0
            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None
        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.
        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])


class BaseAssigner(object):
    """Base assigner that assigns boxes to ground truth boxes."""

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class AlignOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.
    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(self, center_radius=2.5, candidate_topk=10, iou_weight=3.0, cls_weight=1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-07):
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        try:
            assign_result = self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore, eps)
            return assign_result
        except RuntimeError:
            origin_device = pred_scores.device
            warnings.warn('OOM RuntimeError is raised due to the huge memory cost during label assignment. CPU mode is applied in this batch. If you want to avoid this issue, try to reduce the batch size or image size.')
            torch.cuda.empty_cache()
            pred_scores = pred_scores.cpu()
            priors = priors.cpu()
            decoded_bboxes = decoded_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt_labels.cpu()
            assign_result = self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore, eps)
            assign_result.gt_inds = assign_result.gt_inds
            assign_result.max_overlaps = assign_result.max_overlaps
            assign_result.labels = assign_result.labels
            return assign_result

    def _assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-07):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)
        gt_onehot_label = F.one_hot(gt_labels, pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores
        cls_cost = F.binary_cross_entropy(valid_pred_scores, soft_label, reduction='none') * scale_factor.abs().pow(2.0)
        cls_cost = cls_cost.sum(dim=-1)
        cost_matrix = cls_cost * self.cls_weight + iou_cost * self.iou_weight + ~is_in_boxes_and_center * INF
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y
        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y
        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y
        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all
        is_in_boxes_and_centers = is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


@weighted_loss
def giou_loss(pred, target, eps=1e-07):
    """`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


class GIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    """Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = func(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction='mean', loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * quality_focal_loss(pred, target, weight, beta=self.beta, use_sigmoid=self.use_sigmoid, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains             a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


FLIP_LEFT_RIGHT = 0


FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box,
    such as labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format(bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError('last dimension of bbox should have a size of 4, got {}'.format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 0
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 0
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h - TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 0
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 0
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 0
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError('Should not be here')
        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def multiclass_nms(multi_bboxes, multi_scores, score_thr, iou_thr, max_num=100, score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels             are 0-based.
    """
    num_classes = multi_scores.size(1)
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores
    valid_mask = scores > score_thr
    bboxes = torch.masked_select(bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]
    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0,))
        return bboxes, scores, labels
    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)
    if max_num > 0:
        keep = keep[:max_num]
    return bboxes[keep], scores[keep], labels[keep]


def postprocess(cls_scores, bbox_preds, num_classes, conf_thre=0.7, nms_thre=0.45, imgs=None):
    batch_size = bbox_preds.size(0)
    output = [None for _ in range(batch_size)]
    for i in range(batch_size):
        if not bbox_preds[i].size(0):
            continue
        detections, scores, labels = multiclass_nms(bbox_preds[i], cls_scores[i], conf_thre, nms_thre, 500)
        detections = torch.cat((detections, torch.ones_like(scores[:, None]), scores[:, None], labels[:, None]), dim=1)
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    for i in range(len(output)):
        res = output[i]
        if res is None or imgs is None:
            boxlist = BoxList(torch.zeros(0, 4), (0, 0), mode='xyxy')
            boxlist.add_field('objectness', 0)
            boxlist.add_field('scores', 0)
            boxlist.add_field('labels', -1)
        else:
            img_h, img_w = imgs.image_sizes[i]
            boxlist = BoxList(res[:, :4], (img_w, img_h), mode='xyxy')
            boxlist.add_field('objectness', res[:, 4])
            boxlist.add_field('scores', res[:, 5])
            boxlist.add_field('labels', res[:, 6] + 1)
        output[i] = boxlist
    return output


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class ZeroHead(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, feat_channels=256, reg_max=12, strides=[8, 16, 32], norm='gn', act='relu', nms_conf_thre=0.05, nms_iou_thre=0.7, nms=True, **kwargs):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.act = act
        self.strides = strides
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.strides)
        self.cls_out_channels = num_classes + 1
        self.reg_max = reg_max
        self.nms = nms
        self.nms_conf_thre = nms_conf_thre
        self.nms_iou_thre = nms_iou_thre
        self.assigner = AlignOTAAssigner(center_radius=2.5, cls_weight=1.0, iou_weight=3.0)
        self.feat_size = [torch.zeros(4) for _ in strides]
        super(ZeroHead, self).__init__()
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)
        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            cls_convs.append(ConvBNAct(chn, feat_channels, kernel_size, stride=1, groups=1, norm='bn', act=self.act))
            reg_convs.append(ConvBNAct(chn, feat_channels, kernel_size, stride=1, groups=1, norm='bn', act=self.act))
        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        self.gfl_cls = nn.ModuleList([nn.Conv2d(self.feat_channels[i], self.cls_out_channels, 3, padding=1) for i in range(len(self.strides))])
        self.gfl_reg = nn.ModuleList([nn.Conv2d(self.feat_channels[i], 4 * (self.reg_max + 1), 3, padding=1) for i in range(len(self.strides))])
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        if self.training:
            return self.forward_train(xin=xin, labels=labels, imgs=imgs)
        else:
            return self.forward_eval(xin=xin, labels=labels, imgs=imgs)

    def forward_train(self, xin, labels=None, imgs=None, aux_targets=None):
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels') - 1).long())
        mlvl_priors_list = [self.get_single_level_center_priors(xin[i].shape[0], xin[i].shape[-2:], stride, dtype=torch.float32, device=xin[0].device) for i, stride in enumerate(self.strides)]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(self.forward_single, xin, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg, self.scales)
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)
        loss = self.loss(cls_scores, bbox_preds, bbox_before_softmax, gt_bbox_list, gt_cls_list, mlvl_priors)
        return loss

    def forward_eval(self, xin, labels=None, imgs=None):
        if self.feat_size[0] != xin[0].shape:
            mlvl_priors_list = [self.get_single_level_center_priors(xin[i].shape[0], xin[i].shape[-2:], stride, dtype=torch.float32, device=xin[0].device) for i, stride in enumerate(self.strides)]
            self.mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
            self.feat_size[0] = xin[0].shape
        cls_scores, bbox_preds = multi_apply(self.forward_single, xin, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg, self.scales)
        cls_scores = torch.cat(cls_scores, dim=1)[:, :, :self.num_classes]
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_preds = self.integral(bbox_preds) * self.mlvl_priors[..., 2, None]
        bbox_preds = distance2bbox(self.mlvl_priors[..., :2], bbox_preds)
        if self.nms:
            output = postprocess(cls_scores, bbox_preds, self.num_classes, self.nms_conf_thre, self.nms_iou_thre, imgs)
            return output
        return cls_scores, bbox_preds

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x
        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)
        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        if self.training:
            bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)
            bbox_before_softmax = bbox_before_softmax.flatten(start_dim=3).permute(0, 3, 1, 2)
        bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        cls_score = gfl_cls(cls_feat).sigmoid()
        cls_score = cls_score.flatten(start_dim=2).permute(0, 2, 1)
        bbox_pred = bbox_pred.flatten(start_dim=3).permute(0, 3, 1, 2)
        if self.training:
            return cls_score, bbox_pred, bbox_before_softmax
        else:
            return cls_score, bbox_pred

    def get_single_level_center_priors(self, batch_size, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0, int(w), dtype=dtype, device=device) * stride
        y_range = torch.arange(0, int(h), dtype=dtype, device=device) * stride
        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(self, cls_scores, bbox_preds, bbox_before_softmax, gt_bboxes, gt_labels, mlvl_center_priors, gt_bboxes_ignore=None):
        """Compute losses of the head.

        """
        device = cls_scores[0].device
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores, decoded_bboxes, gt_bboxes, mlvl_center_priors, gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None
        labels_list, label_scores_list, label_weights_list, bbox_targets_list, bbox_weights_list, dfl_targets_list, num_pos = cls_reg_targets
        num_total_pos = max(reduce_mean(torch.tensor(num_pos).type(torch.float)).item(), 1.0)
        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_before_softmax = bbox_before_softmax.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores), avg_factor=num_total_pos)
        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes), as_tuple=False).squeeze(1)
        weight_targets = cls_scores.detach()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(decoded_bboxes[pos_inds], bbox_targets[pos_inds], weight=weight_targets, avg_factor=1.0 * norm_factor)
            loss_dfl = self.loss_dfl(bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1), dfl_targets[pos_inds].reshape(-1), weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0 * norm_factor)
        else:
            loss_bbox = bbox_preds.sum() / norm_factor * 0.0
            loss_dfl = bbox_preds.sum() / norm_factor * 0.0
            logger.info(f'No Positive Samples on {bbox_preds.device}! May cause performance decrease. loss_bbox:{loss_bbox:.3f}, loss_dfl:{loss_dfl:.3f}, loss_qfl:{loss_qfl:.3f} ')
        total_loss = loss_qfl + loss_bbox + loss_dfl
        return dict(total_loss=total_loss, loss_cls=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    def get_targets(self, cls_scores, bbox_preds, gt_bboxes_list, mlvl_center_priors, gt_labels_list=None, unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_scores, all_label_weights, all_bbox_targets, all_bbox_weights, all_dfl_targets, all_pos_num = multi_apply(self.get_target_single, mlvl_center_priors, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list)
        if any([(labels is None) for labels in all_labels]):
            return None
        all_pos_num = sum(all_pos_num)
        return all_labels, all_label_scores, all_label_weights, all_bbox_targets, all_bbox_weights, all_dfl_targets, all_pos_num

    def get_target_single(self, center_priors, cls_scores, bbox_preds, gt_bboxes, gt_labels, unmap_outputs=True, gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        num_valid_center = center_priors.shape[0]
        labels = center_priors.new_full((num_valid_center,), self.num_classes, dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)
        if gt_labels.size(0) == 0:
            return labels, label_scores, label_weights, bbox_targets, bbox_weights, dfl_targets, 0
        assign_result = self.assigner.assign(cls_scores.detach(), center_priors, bbox_preds.detach(), gt_bboxes, gt_labels)
        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]
        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = bbox2distance(center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2], pos_bbox_targets / center_priors[pos_inds, None, 2], self.reg_max)
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return labels, label_scores, label_weights, bbox_targets, bbox_weights, dfl_targets, pos_inds.size(0)

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        if gt_bboxes.numel() == 0:
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) - softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * self.tau ** 2
            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss


class MGDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, alpha_mgd=2e-05, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.generation = [nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, kernel_size=3, padding=1)) for channel in channels_t]

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W))
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class MimicLoss(nn.Module):

    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss


class FeatureLoss(nn.Module):

    def __init__(self, channels_s, channels_t, distiller='cwd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0) for channel, tea_channel in zip(channels_s, channels_t)])
        self.norm = [nn.BatchNorm2d(tea_channel, affine=False) for tea_channel in channels_t]
        if distiller == 'mimic':
            self.feature_loss = MimicLoss(channels_s, channels_t)
        elif distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)
        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class GiraffeNeckV2(nn.Module):

    def __init__(self, depth=1.0, hidden_ratio=1.0, in_features=[2, 3, 4], in_channels=[256, 512, 1024], out_channels=[256, 512, 1024], act='silu', spp=False, block_name='BasicBlock'):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = ConvBNAct
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bu_conv13 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.merge_3 = CSPStage(block_name, in_channels[1] + in_channels[2], hidden_ratio, in_channels[2], round(3 * depth), act=act, spp=spp)
        self.bu_conv24 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.merge_4 = CSPStage(block_name, in_channels[0] + in_channels[1] + in_channels[2], hidden_ratio, in_channels[1], round(3 * depth), act=act, spp=spp)
        self.merge_5 = CSPStage(block_name, in_channels[1] + in_channels[0], hidden_ratio, out_channels[0], round(3 * depth), act=act, spp=spp)
        self.bu_conv57 = Conv(out_channels[0], out_channels[0], 3, 2, act=act)
        self.merge_7 = CSPStage(block_name, out_channels[0] + in_channels[1], hidden_ratio, out_channels[1], round(3 * depth), act=act, spp=spp)
        self.bu_conv46 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.bu_conv76 = Conv(out_channels[1], out_channels[1], 3, 2, act=act)
        self.merge_6 = CSPStage(block_name, in_channels[1] + out_channels[1] + in_channels[2], hidden_ratio, out_channels[2], round(3 * depth), act=act, spp=spp)

    def init_weights(self):
        pass

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        [x2, x1, x0] = out_features
        x13 = self.bu_conv13(x1)
        x3 = torch.cat([x0, x13], 1)
        x3 = self.merge_3(x3)
        x34 = self.upsample(x3)
        x24 = self.bu_conv24(x2)
        x4 = torch.cat([x1, x24, x34], 1)
        x4 = self.merge_4(x4)
        x45 = self.upsample(x4)
        x5 = torch.cat([x2, x45], 1)
        x5 = self.merge_5(x5)
        x57 = self.bu_conv57(x5)
        x7 = torch.cat([x4, x57], 1)
        x7 = self.merge_7(x7)
        x46 = self.bu_conv46(x4)
        x76 = self.bu_conv76(x7)
        x6 = torch.cat([x3, x46, x76], 1)
        x6 = self.merge_6(x6)
        outputs = x5, x7, x6
        return outputs


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    else:
        None


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    else:
        raise NotImplementedError


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'GiraffeNeckV2':
        return GiraffeNeckV2(**neck_cfg)
    else:
        raise NotImplementedError


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes, pad_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.pad_sizes = pad_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors
        return ImageList(cast_tensor, self.image_sizes, self.pad_sizes)


def to_image_list(tensors, size_divisible=0, max_size=None):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]
    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        if max_size is None:
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        if size_divisible > 0:
            import math
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        image_sizes = [im.shape[-2:] for im in tensors]
        pad_sizes = [batched_imgs.shape[-2:] for im in batched_imgs]
        return ImageList(batched_imgs, image_sizes, pad_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(type(tensors)))


class Detector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.model.backbone)
        self.neck = build_neck(config.model.neck)
        self.head = build_head(config.model.head)
        self.config = config

    def init_bn(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03

    def init_model(self):
        self.apply(self.init_bn)
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def load_pretrain_detector(self, pretrain_model):
        state_dict = torch.load(pretrain_model, map_location='cpu')['model']
        logger.info(f'Finetune from {pretrain_model}................')
        new_state_dict = {}
        for k, v in self.state_dict().items():
            k = k.replace('module.', '')
            if 'head' in k:
                new_state_dict[k] = self.state_dict()[k]
                continue
            new_state_dict[k] = state_dict[k]
        self.load_state_dict(new_state_dict, strict=True)

    def forward(self, x, targets=None, tea=False, stu=False):
        images = to_image_list(x)
        feature_outs = self.backbone(images.tensors)
        fpn_outs = self.neck(feature_outs)
        if tea:
            return fpn_outs
        else:
            outputs = self.head(fpn_outs, targets, imgs=images)
            if stu:
                return outputs, fpn_outs
            else:
                return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock_3x3_Reverse,
     lambda: ([], {'ch_in': 4, 'ch_hidden_ratio': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvKXBN,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvKXBNRELU,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (End2End,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MimicLoss,
     lambda: ([], {'channels_s': 4, 'channels_t': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ONNX_TRT7,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ONNX_TRT8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RepConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResConvBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'btn_c': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (SPPBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SuperResStem,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'btn_c': 4, 'kernel_size': 4, 'stride': 1, 'num_blocks': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
]

class Test_tinyvision_DAMO_YOLO(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

